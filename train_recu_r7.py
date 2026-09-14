import argparse, json, time
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn

from recu_hw.r6 import (
    R6ResNet20,
    is_power_of_two_tensor,
    parameter_freeze_report,
    stem_binary_weight_stats,
)
from recu_hw.r5t_data import build_r5t_loaders
from recu_hw.layers import iter_recu_binary_convs
from recu_hw.r7 import build_r7_from_r6, exact_head_fold_max_abs_error
from recu_hw.utils import load_json, save_json, seed_everything, git_commit


def set_tau_all(model,tau):
    for m in iter_recu_binary_convs(model):
        m.set_tau(tau)


@torch.no_grad()
def evaluate(model,loader,criterion,device):
    model.eval()
    total=correct=0
    loss_sum=0.0
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        logits=model(x)
        loss=criterion(logits,y)
        loss_sum += loss.item()*y.size(0)
        correct += (logits.argmax(1)==y).sum().item()
        total += y.size(0)
    return loss_sum/max(total,1),100.0*correct/max(total,1)


def load_source(path,device,resolution=8):
    state=torch.load(path,map_location=device)
    stored_acc=float(state.get("best_acc", float("nan")))
    stored_epoch=int(state.get("best_epoch", -1))
    if abs(stored_acc - 85.52) > 0.20 or stored_epoch != 97:
        raise RuntimeError(
            "R7 source is not the required R6 best checkpoint: "
            f"best_acc={stored_acc}, best_epoch={stored_epoch}"
        )
    m=R6ResNet20(num_classes=10,resolution=resolution).to(device)
    m.load_state_dict(state["model"],strict=True)
    m.conv1.set_binary_weight(True)
    m.stem_affine.quantize_k=True
    for module in m.modules():
        if hasattr(module, "alpha"):
            module.alpha.requires_grad_(False)
    stem=stem_binary_weight_stats(m)
    if not stem["is_strict_binary"] or not stem["conv_bias_is_none"]:
        raise RuntimeError(f"R7 source stem invariant failed: {stem}")
    if not is_power_of_two_tensor(m.stem_affine.effective_k()):
        raise RuntimeError("R7 source stem K is not signed power-of-two")
    if hasattr(m,"bn1"):
        raise RuntimeError("R7 source unexpectedly contains standalone stem bn1")
    m.eval()
    return m,state


def assert_finite_and_frozen(model, frozen_snapshot):
    if hasattr(model,"bn2"):
        raise RuntimeError("R7 must not contain standalone bn2")
    if model.linear.bias is None:
        raise RuntimeError("R7 final FC bias was removed")
    if not all(bool(torch.isfinite(p).all().item()) for p in model.parameters()):
        raise RuntimeError("R7 model contains NaN/Inf parameters")
    if not bool(torch.isfinite(model.stem_affine.k_latent).all().item()):
        raise RuntimeError("R7 stem latent K contains NaN/Inf")
    if not bool(torch.isfinite(model.stem_affine.bias).all().item()):
        raise RuntimeError("R7 stem B contains NaN/Inf")
    if not bool(torch.isfinite(model.head_affine.k_latent).all().item()):
        raise RuntimeError("R7 head latent K contains NaN/Inf")
    if not bool(torch.isfinite(model.head_affine.bias).all().item()):
        raise RuntimeError("R7 head B contains NaN/Inf")
    current=dict(model.named_parameters())
    for name,value in frozen_snapshot.items():
        if current[name].requires_grad:
            raise RuntimeError(f"Frozen parameter was unfrozen: {name}")
        if not torch.equal(current[name].detach(),value):
            raise RuntimeError(f"Frozen parameter changed: {name}")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    ap.add_argument("--source-r6-checkpoint",required=True)
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--diagnostic-epochs",type=int,default=None)
    ap.add_argument("--lr",type=float,default=None,
                    help="Optional controlled LR override after the BN-removal stability probe")
    ap.add_argument("--device",default=None)
    args=ap.parse_args()

    cfg=load_json(args.config)
    seed_everything(int(cfg["experiment"].get("seed",123)))
    device=torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    resolution=int(cfg["model"].get("resolution",8))

    src,src_state=load_source(args.source_r6_checkpoint,device,resolution)
    r7=build_r7_from_r6(
        src,
        resolution=resolution,
        head_min_exp=cfg["model"].get("head_min_exp"),
        head_max_exp=cfg["model"].get("head_max_exp"),
    ).to(device)

    freeze_report=parameter_freeze_report(r7)
    if freeze_report["frozen_params"] != 672:
        raise RuntimeError(
            "Unexpected frozen parameter count: "
            f"{freeze_report['frozen_params']} (expected 672 R4 compatibility alpha parameters)"
        )
    if set(freeze_report["frozen_parameter_categories"]) != {"r4_compatibility_alpha"}:
        raise RuntimeError(
            "Unexpected frozen parameter categories: "
            f"{freeze_report['frozen_parameter_categories']}"
        )
    frozen_snapshot={
        name:p.detach().clone()
        for name,p in r7.named_parameters()
        if not p.requires_grad
    }
    assert_finite_and_frozen(r7,frozen_snapshot)

    tau=float(cfg["finetune"].get("tau",0.99))
    set_tau_all(r7,tau)

    train_loader,test_loader=build_r5t_loaders(
        {"dataset":cfg["dataset"],"training":cfg["training"]},smoke=args.smoke
    )
    criterion=nn.CrossEntropyLoss()

    max_err=0.0
    for i,(x,_) in enumerate(test_loader):
        if i>=int(cfg["verification"].get("max_fold_batches",8)): break
        max_err=max(max_err,exact_head_fold_max_abs_error(src,r7,x.to(device)))

    tol=float(cfg["verification"].get("fold_tolerance",1e-5))
    if max_err>=tol:
        raise RuntimeError(f"Exact head fold failed: {max_err:.8e} >= {tol:.8e}")

    _,src_acc=evaluate(src,test_loader,criterion,device)
    r7.head_affine.quantize_k=True
    _,zero_acc=evaluate(r7,test_loader,criterion,device)

    ft=cfg["finetune"]
    full_epochs=int(ft.get("epochs",100))
    epochs=1 if args.smoke else (
        int(args.diagnostic_epochs) if args.diagnostic_epochs is not None else full_epochs
    )

    effective_lr=float(args.lr) if args.lr is not None else float(ft.get("lr",1e-3))
    opt=torch.optim.SGD(
        [p for p in r7.parameters() if p.requires_grad],
        lr=effective_lr,
        momentum=float(ft.get("momentum",0.9)),
        weight_decay=float(ft.get("weight_decay",0.0)),
    )
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(1,full_epochs))

    out_root=Path(cfg["output"]["root"]); out_root.mkdir(parents=True,exist_ok=True)
    run_dir=out_root/f"{cfg['experiment']['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if args.smoke: run_dir=Path(str(run_dir)+"_smoke")
    run_dir.mkdir(parents=True,exist_ok=False)

    save_json(cfg,run_dir/"config.json")
    save_json({
        "source_checkpoint":args.source_r6_checkpoint,
        "source_stored_best_acc":src_state.get("best_acc"),
        "source_stored_best_epoch":src_state.get("best_epoch"),
        "source_reload_acc":src_acc,
        "exact_head_fold_max_abs_error":max_err,
        "zero_epoch_hard_projection_acc":zero_acc,
        "delta_projection_pp":zero_acc-src_acc,
        "initial_head_affine":r7.head_affine.stats(),
        "stem_affine":r7.stem_affine.stats(),
        "stem_binary":stem_binary_weight_stats(r7),
        "trainable_params":freeze_report["trainable_params"],
        "frozen_params":freeze_report["frozen_params"],
        "frozen_parameter_categories":freeze_report["frozen_parameter_categories"],
        "frozen_parameter_names":freeze_report["frozen_parameter_names"],
        "linear_bias_present":r7.linear.bias is not None,
        "effective_lr":effective_lr,
    },run_dir/"conversion.json")

    best_acc=zero_acc; best_epoch=0
    best_path=run_dir/"best.pt"; last_path=run_dir/"last.pt"
    history=[]

    torch.save({
        "epoch":0,"model":r7.state_dict(),"best_acc":best_acc,"best_epoch":0,
        "config":cfg,"source_r6_checkpoint":args.source_r6_checkpoint,
        "head_affine_quantized":True
    },best_path)

    t0=time.time()

    for epoch in range(epochs):
        r7.train(); set_tau_all(r7,tau)
        total=correct=0; loss_sum=0.0

        for x,y in train_loader:
            x,y=x.to(device),y.to(device)
            opt.zero_grad(set_to_none=True)
            logits=r7(x)
            loss=criterion(logits,y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch+1}")
            loss.backward(); opt.step()
            loss_sum += loss.item()*y.size(0)
            correct += (logits.argmax(1)==y).sum().item()
            total += y.size(0)

        sched.step()
        test_loss,test_acc=evaluate(r7,test_loader,criterion,device)
        assert_finite_and_frozen(r7,frozen_snapshot)
        head=r7.head_affine.stats()
        stem=r7.stem_affine.stats()
        sb=stem_binary_weight_stats(r7)

        if not head["is_exact_signed_pow2"]:
            raise RuntimeError("Head Khat is not signed power-of-two")
        if not stem["is_exact_signed_pow2"]:
            raise RuntimeError("Stem Khat changed away from signed power-of-two")
        if not sb["is_strict_binary"]:
            raise RuntimeError("Stem effective weight changed away from {-1,+1}")
        if r7.linear.bias is None:
            raise RuntimeError("Final FC bias disappeared")

        row={
            "epoch":epoch+1,"lr":opt.param_groups[0]["lr"],
            "train_loss":loss_sum/max(total,1),
            "train_acc":100.0*correct/max(total,1),
            "test_loss":test_loss,"test_acc":test_acc,
            "head_affine":head,"stem_affine":stem,"stem_binary":sb,
        }
        history.append(row); save_json(history,run_dir/"history.json")

        if test_acc>best_acc:
            best_acc=test_acc; best_epoch=epoch+1
            torch.save({
                "epoch":epoch+1,"model":r7.state_dict(),
                "best_acc":best_acc,"best_epoch":best_epoch,"config":cfg,
                "source_r6_checkpoint":args.source_r6_checkpoint,
                "head_affine_quantized":True
            },best_path)

        torch.save({
            "epoch":epoch+1,"model":r7.state_dict(),
            "best_acc":best_acc,"best_epoch":best_epoch,"config":cfg,
            "source_r6_checkpoint":args.source_r6_checkpoint,
            "head_affine_quantized":True
        },last_path)

        print(f"epoch={epoch+1}/{epochs} test={test_acc:.2f}% best={best_acc:.2f}% head_exp={head['exponent_histogram']}")

    elapsed=time.time()-t0
    bs=torch.load(best_path,map_location=device)
    r7.load_state_dict(bs["model"],strict=True)
    r7.head_affine.quantize_k=True
    r7.stem_affine.quantize_k=True
    _,reload_acc=evaluate(r7,test_loader,criterion,device)

    final_head=r7.head_affine.stats()
    final_stem=r7.stem_affine.stats()
    final_sb=stem_binary_weight_stats(r7)
    assert_finite_and_frozen(r7,frozen_snapshot)

    if not final_head["is_exact_signed_pow2"]:
        raise RuntimeError("Reloaded head Khat not signed power-of-two")
    if not final_stem["is_exact_signed_pow2"]:
        raise RuntimeError("Reloaded stem Khat not signed power-of-two")
    if not final_sb["is_strict_binary"]:
        raise RuntimeError("Reloaded stem weight not {-1,+1}")
    if not final_sb["conv_bias_is_none"]:
        raise RuntimeError("Reloaded stem Conv unexpectedly has bias")
    if hasattr(r7,"bn2"):
        raise RuntimeError("R7 must not contain standalone bn2")
    if r7.linear.bias is None:
        raise RuntimeError("Reloaded final FC bias is missing")

    summary={
        "source_acc":src_acc,
        "exact_head_fold_max_abs_error":max_err,
        "zero_epoch_hard_projection_acc":zero_acc,
        "delta_zero_epoch_vs_source_pp":zero_acc-src_acc,
        "best_acc":best_acc,
        "best_epoch":best_epoch,
        "final_acc":history[-1]["test_acc"] if history else zero_acc,
        "reload_acc":reload_acc,
        "delta_best_vs_source_pp":best_acc-src_acc,
        "elapsed_s":elapsed,
        "head_affine":final_head,
        "stem_affine":final_stem,
        "stem_binary":final_sb,
        "trainable_params":freeze_report["trainable_params"],
        "frozen_params":freeze_report["frozen_params"],
        "frozen_parameter_categories":freeze_report["frozen_parameter_categories"],
        "frozen_parameter_names":freeze_report["frozen_parameter_names"],
        "bn2_removed":not hasattr(r7,"bn2"),
        "linear_bias_present":r7.linear.bias is not None,
        "effective_lr":effective_lr,
        "git_commit":git_commit(),
    }
    save_json(summary,run_dir/"summary.json")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
