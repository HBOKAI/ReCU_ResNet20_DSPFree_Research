import argparse, json, sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from recu_hw.r5t import R5TResNet20
from recu_hw.r5t_data import build_r5t_loaders
from recu_hw.r6 import (
    R6ResNet20,
    build_r6_from_r5t,
    exact_fold_max_abs_error,
    is_power_of_two_tensor,
    parameter_freeze_report,
    stem_binary_weight_stats,
)


@torch.no_grad()
def evaluate(model,loader,device):
    model.eval()
    total=correct=0
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        p=model(x).argmax(1)
        correct += (p==y).sum().item()
        total += y.size(0)
    return 100.0*correct/max(total,1)


def load_source(path,device,resolution):
    state=torch.load(path,map_location=device)
    stored_acc=float(state.get("best_acc", float("nan")))
    stored_epoch=int(state.get("best_epoch", -1))
    if abs(stored_acc - 85.14) > 0.20 or stored_epoch != 200:
        raise SystemExit(
            "FAIL: wrong R5T-Long source checkpoint: "
            f"best_acc={stored_acc}, best_epoch={stored_epoch}"
        )
    m=R5TResNet20(num_classes=10,resolution=resolution,binary_stem=True).to(device)
    m.load_state_dict(state["model"],strict=True)
    m.conv1.set_binary_weight(True)
    # state_dict does not carry requires_grad; restore the R5T frozen alpha
    # policy before converting to R6.
    for module in m.modules():
        if hasattr(module, "alpha"):
            module.alpha.requires_grad_(False)
    stem=stem_binary_weight_stats(m)
    if not stem["is_strict_binary"] or not stem["conv_bias_is_none"]:
        raise SystemExit(f"FAIL: source stem invariant failed: {stem}")
    m.eval()
    return m,state


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source-r5t-long-checkpoint",required=True)
    ap.add_argument("--resolution",type=int,default=8)
    ap.add_argument("--device",default=None)
    ap.add_argument("--max-fold-batches",type=int,default=8)
    ap.add_argument("--r6-checkpoint",default=None)
    args=ap.parse_args()

    device=torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    src,state=load_source(args.source_r5t_long_checkpoint,device,args.resolution)
    r6=build_r6_from_r5t(src,resolution=args.resolution).to(device).eval()

    cfg={"dataset":{"name":"cifar10","root":"./data","download":True},
         "training":{"batch_size":128,"test_batch_size":128,"num_workers":0,"windows_safe_smoke":True}}
    _,test_loader=build_r5t_loaders(cfg,smoke=False)

    max_err=0.0
    for i,(x,_) in enumerate(test_loader):
        if i>=args.max_fold_batches: break
        max_err=max(max_err,exact_fold_max_abs_error(src,r6,x.to(device)))

    if max_err>=1e-5:
        raise SystemExit(f"FAIL: exact fold error {max_err:.8e} >= 1e-5")

    src_acc=evaluate(src,test_loader,device)
    r6.stem_affine.quantize_k=True
    proj_acc=evaluate(r6,test_loader,device)

    report={
        "source_checkpoint":args.source_r5t_long_checkpoint,
        "stored_best_acc":state.get("best_acc"),
        "stored_best_epoch":state.get("best_epoch"),
        "source_reload_acc":src_acc,
        "exact_fold_max_abs_error":max_err,
        "hard_projection_pow2_acc":proj_acc,
        "delta_projection_pp":proj_acc-src_acc,
        "stem_binary_weight":stem_binary_weight_stats(r6),
        "stem_affine":r6.stem_affine.stats(),
        "r6_parameter_freeze":parameter_freeze_report(r6),
        "bn1_removed_in_r6":not hasattr(r6,"bn1"),
    }

    if args.r6_checkpoint:
        r6_state=torch.load(args.r6_checkpoint,map_location=device)
        reloaded=R6ResNet20(resolution=args.resolution).to(device)
        reloaded.load_state_dict(r6_state["model"],strict=True)
        reloaded.eval()
        reloaded_k=reloaded.stem_affine.effective_k()
        reloaded_stem=stem_binary_weight_stats(reloaded)
        report["r6_reload"]={
            "checkpoint":args.r6_checkpoint,
            "checkpoint_epoch":r6_state.get("epoch"),
            "checkpoint_best_acc":r6_state.get("best_acc"),
            "full_reload_acc":evaluate(reloaded,test_loader,device),
            "k_shape":list(reloaded.stem_affine.k_latent.shape),
            "b_shape":list(reloaded.stem_affine.bias.shape),
            "effective_k_unique":sorted(float(v) for v in torch.unique(reloaded_k).cpu().tolist()),
            "is_exact_signed_pow2":is_power_of_two_tensor(reloaded_k),
            "stem_binary":reloaded_stem,
            "bn1_removed":not hasattr(reloaded,"bn1"),
            "finite":all(bool(torch.isfinite(p).all().item()) for p in reloaded.parameters()),
            "parameter_freeze":parameter_freeze_report(reloaded),
        }
    print(json.dumps(report,indent=2))

    if not report["stem_binary_weight"]["is_strict_binary"]:
        raise SystemExit("FAIL: stem effective weight is not {-1,+1}")
    if not report["stem_binary_weight"]["conv_bias_is_none"]:
        raise SystemExit("FAIL: stem Conv unexpectedly has bias")
    if not report["stem_affine"]["is_exact_signed_pow2"]:
        raise SystemExit("FAIL: Khat is not signed power-of-two")
    if args.r6_checkpoint:
        rr=report["r6_reload"]
        if not rr["is_exact_signed_pow2"] or not rr["stem_binary"]["is_strict_binary"]:
            raise SystemExit("FAIL: reloaded R6 inference invariant failed")
        if not rr["stem_binary"]["conv_bias_is_none"] or not rr["bn1_removed"] or not rr["finite"]:
            raise SystemExit("FAIL: reloaded R6 structure/finite invariant failed")


if __name__=="__main__":
    main()
