import argparse, json, time
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn

from recu_hw.r5t import R5TResNet20
from recu_hw.r5t_data import build_r5t_loaders
from recu_hw.layers import iter_recu_binary_convs
from recu_hw.r6 import (
    build_r6_from_r5t,
    exact_fold_max_abs_error,
    parameter_freeze_report,
    stem_binary_weight_stats,
)
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


def load_source(path,device,resolution):
    state=torch.load(path,map_location=device)
    stored_acc=float(state.get("best_acc", float("nan")))
    stored_epoch=int(state.get("best_epoch", -1))
    if abs(stored_acc - 85.14) > 0.20 or stored_epoch != 200:
        raise RuntimeError(
            "R6 source is not the required R5T-Long T2 checkpoint: "
            f"best_acc={stored_acc}, best_epoch={stored_epoch}"
        )
    m=R5TResNet20(num_classes=10,resolution=resolution,binary_stem=True).to(device)
    m.load_state_dict(state["model"],strict=True)
    m.conv1.set_binary_weight(True)
    # requires_grad is not serialized in state_dict. Reapply the R5T frozen
    # policy before building R6 so the 672 compatibility alpha parameters
    # cannot accidentally enter the R6 optimizer.
    for module in m.modules():
        if hasattr(module, "alpha"):
            module.alpha.requires_grad_(False)
    source_weight= m.conv1.effective_weight()
    source_unique=sorted(float(v) for v in torch.unique(source_weight).detach().cpu().tolist())
    if source_unique != [-1.0, 1.0] or m.conv1.bias is not None:
        raise RuntimeError(
            "R6 source stem invariant failed: "
            f"unique={source_unique}, bias_is_none={m.conv1.bias is None}"
        )
    m.eval()
    return m,state


def assert_finite_and_frozen(model, frozen_snapshot):
    if not all(bool(torch.isfinite(p).all().item()) for p in model.parameters()):
        raise RuntimeError("R6 model contains NaN/Inf parameters")
    if not bool(torch.isfinite(model.stem_affine.k_latent).all().item()):
        raise RuntimeError("R6 latent K contains NaN/Inf")
    if not bool(torch.isfinite(model.stem_affine.bias).all().item()):
        raise RuntimeError("R6 B contains NaN/Inf")
    current = dict(model.named_parameters())
    for name, value in frozen_snapshot.items():
        if current[name].requires_grad:
            raise RuntimeError(f"Frozen parameter was unfrozen: {name}")
        if not torch.equal(current[name].detach(), value):
            raise RuntimeError(f"Frozen parameter changed: {name}")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    ap.add_argument("--source-r5t-long-checkpoint",required=True)
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--diagnostic-epochs",type=int,default=None)
    ap.add_argument("--device",default=None)
    args=ap.parse_args()

    cfg=load_json(args.config)
    seed_everything(int(cfg["experiment"].get("seed",123)))
    device=torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    resolution=int(cfg["model"].get("resolution",8))

    src,src_state=load_source(args.source_r5t_long_checkpoint,device,resolution)
    r6=build_r6_from_r5t(
        src,resolution=resolution,
        min_exp=cfg["model"].get("min_exp"),
        max_exp=cfg["model"].get("max_exp")
    ).to(device)

    freeze_report=parameter_freeze_report(r6)
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
        name: p.detach().clone()
        for name,p in r6.named_parameters()
        if not p.requires_grad
    }
    assert_finite_and_frozen(r6, frozen_snapshot)

    tau=float(cfg["finetune"].get("tau",0.99))
    set_tau_all(r6,tau)

    train_loader,test_loader=build_r5t_loaders(
        {"dataset":cfg["dataset"],"training":cfg["training"]},smoke=args.smoke
    )
    criterion=nn.CrossEntropyLoss()

    max_err=0.0
    for i,(x,_) in enumerate(test_loader):
        if i>=int(cfg["verification"].get("max_fold_batches",8)): break
        max_err=max(max_err,exact_fold_max_abs_error(src,r6,x.to(device)))
    tol=float(cfg["verification"].get("fold_tolerance",1e-5))
    if max_err>=tol:
        raise RuntimeError(f"Exact fold failed: {max_err:.8e} >= {tol:.8e}")

    _,src_acc=evaluate(src,test_loader,criterion,device)
    r6.stem_affine.quantize_k=True
    _,zero_acc=evaluate(r6,test_loader,criterion,device)

    stem=stem_binary_weight_stats(r6)
    if not stem["is_strict_binary"] or not stem["conv_bias_is_none"]:
        raise RuntimeError("Stem binary/bias invariant failed")

    ft=cfg["finetune"]
    full_epochs=int(ft.get("epochs",100))
    epochs=1 if args.smoke else (
        int(args.diagnostic_epochs) if args.diagnostic_epochs is not None else full_epochs
    )

    opt=torch.optim.SGD(
        [p for p in r6.parameters() if p.requires_grad],
        lr=float(ft.get("lr",1e-3)),
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
        "source_checkpoint":args.source_r5t_long_checkpoint,
        "source_stored_best_acc":src_state.get("best_acc"),
        "source_stored_best_epoch":src_state.get("best_epoch"),
        "source_reload_acc":src_acc,
        "exact_fold_max_abs_error":max_err,
        "zero_epoch_hard_projection_acc":zero_acc,
        "delta_projection_pp":zero_acc-src_acc,
        "initial_stem_binary":stem,
        "initial_stem_affine":r6.stem_affine.stats(),
        "trainable_params":sum(p.numel() for p in r6.parameters() if p.requires_grad),
        "frozen_params":sum(p.numel() for p in r6.parameters() if not p.requires_grad),
        "frozen_parameter_categories":freeze_report["frozen_parameter_categories"],
        "frozen_parameter_names":freeze_report["frozen_parameter_names"],
    },run_dir/"conversion.json")

    best_acc=zero_acc; best_epoch=0
    best_path=run_dir/"best.pt"; last_path=run_dir/"last.pt"
    history=[]

    torch.save({
        "epoch":0,"model":r6.state_dict(),"best_acc":best_acc,"best_epoch":0,
        "config":cfg,"source_r5t_long_checkpoint":args.source_r5t_long_checkpoint,
        "stem_affine_quantized":True
    },best_path)

    t0=time.time()

    for epoch in range(epochs):
        r6.train(); set_tau_all(r6,tau)
        total=correct=0; loss_sum=0.0

        for x,y in train_loader:
            x,y=x.to(device),y.to(device)
            opt.zero_grad(set_to_none=True)
            logits=r6(x)
            loss=criterion(logits,y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch+1}")
            loss.backward(); opt.step()
            loss_sum += loss.item()*y.size(0)
            correct += (logits.argmax(1)==y).sum().item()
            total += y.size(0)

        sched.step()
        test_loss,test_acc=evaluate(r6,test_loader,criterion,device)
        aff=r6.stem_affine.stats()
        stem=stem_binary_weight_stats(r6)

        if not aff["is_exact_signed_pow2"]:
            raise RuntimeError("Khat is not signed power-of-two")
        if not stem["is_strict_binary"]:
            raise RuntimeError("Stem effective weights are not {-1,+1}")
        assert_finite_and_frozen(r6, frozen_snapshot)

        row={
            "epoch":epoch+1,"lr":opt.param_groups[0]["lr"],
            "train_loss":loss_sum/max(total,1),
            "train_acc":100.0*correct/max(total,1),
            "test_loss":test_loss,"test_acc":test_acc,
            "stem_affine":aff,"stem_binary":stem,
        }
        history.append(row); save_json(history,run_dir/"history.json")

        if test_acc>best_acc:
            best_acc=test_acc; best_epoch=epoch+1
            torch.save({
                "epoch":epoch+1,"model":r6.state_dict(),
                "best_acc":best_acc,"best_epoch":best_epoch,"config":cfg,
                "source_r5t_long_checkpoint":args.source_r5t_long_checkpoint,
                "stem_affine_quantized":True
            },best_path)

        torch.save({
            "epoch":epoch+1,"model":r6.state_dict(),
            "best_acc":best_acc,"best_epoch":best_epoch,"config":cfg,
            "source_r5t_long_checkpoint":args.source_r5t_long_checkpoint,
            "stem_affine_quantized":True
        },last_path)

        print(f"epoch={epoch+1}/{epochs} test={test_acc:.2f}% best={best_acc:.2f}% exp={aff['exponent_histogram']}")

    elapsed=time.time()-t0
    bs=torch.load(best_path,map_location=device)
    r6.load_state_dict(bs["model"],strict=True)
    r6.stem_affine.quantize_k=True
    _,reload_acc=evaluate(r6,test_loader,criterion,device)

    final_aff=r6.stem_affine.stats()
    final_stem=stem_binary_weight_stats(r6)
    assert_finite_and_frozen(r6, frozen_snapshot)

    if not final_aff["is_exact_signed_pow2"]:
        raise RuntimeError("Reloaded Khat not signed power-of-two")
    if not final_stem["is_strict_binary"]:
        raise RuntimeError("Reloaded stem not {-1,+1}")
    if not final_stem["conv_bias_is_none"]:
        raise RuntimeError("Reloaded stem Conv unexpectedly has bias")
    if hasattr(r6,"bn1"):
        raise RuntimeError("R6 must not have standalone stem bn1")

    summary={
        "source_acc":src_acc,
        "exact_fold_max_abs_error":max_err,
        "zero_epoch_hard_projection_acc":zero_acc,
        "delta_zero_epoch_vs_source_pp":zero_acc-src_acc,
        "best_acc":best_acc,
        "best_epoch":best_epoch,
        "final_acc":history[-1]["test_acc"] if history else zero_acc,
        "reload_acc":reload_acc,
        "delta_best_vs_source_pp":best_acc-src_acc,
        "elapsed_s":elapsed,
        "stem_affine":final_aff,
        "stem_binary":final_stem,
        "trainable_params":freeze_report["trainable_params"],
        "frozen_params":freeze_report["frozen_params"],
        "frozen_parameter_categories":freeze_report["frozen_parameter_categories"],
        "frozen_parameter_names":freeze_report["frozen_parameter_names"],
        "bn1_removed":not hasattr(r6,"bn1"),
        "git_commit":git_commit(),
    }
    save_json(summary,run_dir/"summary.json")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
