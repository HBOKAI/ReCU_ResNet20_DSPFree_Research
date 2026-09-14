import argparse, json, sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from recu_hw.r6 import (
    R6ResNet20,
    is_power_of_two_tensor,
    parameter_freeze_report,
    stem_binary_weight_stats,
)
from recu_hw.r5t_data import build_r5t_loaders
from recu_hw.r7 import R7ResNet20, build_r7_from_r6, exact_head_fold_max_abs_error


@torch.no_grad()
def evaluate(model,loader,device):
    model.eval()
    total=correct=0
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        pred=model(x).argmax(1)
        correct += (pred==y).sum().item()
        total += y.size(0)
    return 100.0*correct/max(total,1)


def load_source(path,device,resolution=8):
    state=torch.load(path,map_location=device)
    stored_acc=float(state.get("best_acc", float("nan")))
    stored_epoch=int(state.get("best_epoch", -1))
    if abs(stored_acc - 85.52) > 0.20 or stored_epoch != 97:
        raise SystemExit(
            "FAIL: wrong R6 source checkpoint: "
            f"best_acc={stored_acc}, best_epoch={stored_epoch}"
        )
    m=R6ResNet20(num_classes=10,resolution=resolution).to(device)
    m.load_state_dict(state["model"],strict=True)
    m.conv1.set_binary_weight(True)
    m.stem_affine.quantize_k=True
    for module in m.modules():
        if hasattr(module,"alpha"):
            module.alpha.requires_grad_(False)
    stem=stem_binary_weight_stats(m)
    if not stem["is_strict_binary"] or not stem["conv_bias_is_none"]:
        raise SystemExit(f"FAIL: source stem invariant failed: {stem}")
    if not is_power_of_two_tensor(m.stem_affine.effective_k()):
        raise SystemExit("FAIL: source stem K is not signed power-of-two")
    if hasattr(m,"bn1"):
        raise SystemExit("FAIL: source R6 unexpectedly has standalone bn1")
    m.eval()
    return m,state


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source-r6-checkpoint",required=True)
    ap.add_argument("--resolution",type=int,default=8)
    ap.add_argument("--device",default=None)
    ap.add_argument("--max-fold-batches",type=int,default=8)
    ap.add_argument("--r7-checkpoint",default=None)
    args=ap.parse_args()

    device=torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    src,state=load_source(args.source_r6_checkpoint,device,args.resolution)
    r7=build_r7_from_r6(src,resolution=args.resolution).to(device).eval()

    cfg={"dataset":{"name":"cifar10","root":"./data","download":True},
         "training":{"batch_size":128,"test_batch_size":128,"num_workers":0,"windows_safe_smoke":True}}
    _,test_loader=build_r5t_loaders(cfg,smoke=False)

    max_err=0.0
    for i,(x,_) in enumerate(test_loader):
        if i>=args.max_fold_batches: break
        max_err=max(max_err,exact_head_fold_max_abs_error(src,r7,x.to(device)))

    if max_err>=1e-5:
        raise SystemExit(f"FAIL: exact head BN fold error {max_err:.8e} >= 1e-5")

    src_acc=evaluate(src,test_loader,device)
    r7.head_affine.quantize_k=True
    proj_acc=evaluate(r7,test_loader,device)

    report={
        "source_checkpoint":args.source_r6_checkpoint,
        "stored_best_acc":state.get("best_acc"),
        "stored_best_epoch":state.get("best_epoch"),
        "source_reload_acc":src_acc,
        "exact_head_fold_max_abs_error":max_err,
        "hard_projection_pow2_acc":proj_acc,
        "delta_projection_pp":proj_acc-src_acc,
        "stem_binary_weight":stem_binary_weight_stats(r7),
        "stem_affine":r7.stem_affine.stats(),
        "head_affine":r7.head_affine.stats(),
        "r7_parameter_freeze":parameter_freeze_report(r7),
        "bn2_removed_in_r7":not hasattr(r7,"bn2"),
        "linear_bias_present":r7.linear.bias is not None,
    }

    if args.r7_checkpoint:
        r7_state=torch.load(args.r7_checkpoint,map_location=device)
        reloaded=R7ResNet20(resolution=args.resolution).to(device)
        reloaded.load_state_dict(r7_state["model"],strict=True)
        reloaded.eval()
        head_k=reloaded.head_affine.effective_k()
        stem_k=reloaded.stem_affine.effective_k()
        reloaded_stem=stem_binary_weight_stats(reloaded)
        report["r7_reload"]={
            "checkpoint":args.r7_checkpoint,
            "checkpoint_epoch":r7_state.get("epoch"),
            "checkpoint_best_acc":r7_state.get("best_acc"),
            "full_reload_acc":evaluate(reloaded,test_loader,device),
            "head_k_shape":list(reloaded.head_affine.k_latent.shape),
            "head_b_shape":list(reloaded.head_affine.bias.shape),
            "head_effective_k_unique":sorted(float(v) for v in torch.unique(head_k).cpu().tolist()),
            "head_is_exact_signed_pow2":is_power_of_two_tensor(head_k),
            "stem_is_exact_signed_pow2":is_power_of_two_tensor(stem_k),
            "stem_binary":reloaded_stem,
            "bn2_removed":not hasattr(reloaded,"bn2"),
            "linear_bias_present":reloaded.linear.bias is not None,
            "finite":all(bool(torch.isfinite(p).all().item()) for p in reloaded.parameters()),
            "parameter_freeze":parameter_freeze_report(reloaded),
        }
    print(json.dumps(report,indent=2))

    if not report["stem_binary_weight"]["is_strict_binary"]:
        raise SystemExit("FAIL: stem effective weight is not {-1,+1}")
    if not report["stem_affine"]["is_exact_signed_pow2"]:
        raise SystemExit("FAIL: stem Khat is not signed power-of-two")
    if not report["head_affine"]["is_exact_signed_pow2"]:
        raise SystemExit("FAIL: head Khat is not signed power-of-two")
    if args.r7_checkpoint:
        rr=report["r7_reload"]
        if not rr["head_is_exact_signed_pow2"] or not rr["stem_is_exact_signed_pow2"]:
            raise SystemExit("FAIL: reloaded R7 K invariant failed")
        if not rr["stem_binary"]["is_strict_binary"] or not rr["stem_binary"]["conv_bias_is_none"]:
            raise SystemExit("FAIL: reloaded R7 stem invariant failed")
        if not rr["bn2_removed"] or not rr["linear_bias_present"] or not rr["finite"]:
            raise SystemExit("FAIL: reloaded R7 structure/finite invariant failed")


if __name__=="__main__":
    main()
