import argparse, json, time
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn

from recu_hw.data import build_recu_loaders
from recu_hw.layers import iter_recu_binary_convs
from recu_hw.r4 import R4ResNet20
from recu_hw.r5ab import build_r5ab_from_r4, progressive_lambda, stem_stats, folded_stem_bn_params
from recu_hw.utils import git_commit, load_json, save_json, seed_everything


@torch.no_grad()
def evaluate(model, loader, criterion, device, force_lambda=None):
    old = model.conv1.progress_lambda
    if force_lambda is not None:
        model.conv1.set_progress_lambda(force_lambda)
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    if force_lambda is not None:
        model.conv1.set_progress_lambda(old)
    return loss_sum/max(total,1), 100.0*correct/max(total,1)


def set_tau_all(model, tau):
    for m in iter_recu_binary_convs(model):
        m.set_tau(tau)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--source-r4-checkpoint", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--diagnostic-epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_json(args.config)
    seed_everything(int(cfg["experiment"].get("seed",123)))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    src = torch.load(args.source_r4_checkpoint, map_location=device)
    r4 = R4ResNet20(num_classes=10).to(device)
    r4.load_state_dict(src["model"], strict=True)
    r4.eval()

    model = build_r5ab_from_r4(
        r4,
        input_scale_exp=int(cfg["model"].get("input_scale_exp",-5))
    ).to(device)

    tau = float(cfg["finetune"].get("tau",0.99))
    set_tau_all(model, tau)

    train_loader, test_loader = build_recu_loaders(
        {"dataset": cfg["dataset"], "training": cfg["training"]},
        smoke=args.smoke
    )
    criterion = nn.CrossEntropyLoss()

    _, acc_l0 = evaluate(model,test_loader,criterion,device,force_lambda=0.0)
    _, acc_l1 = evaluate(model,test_loader,criterion,device,force_lambda=1.0)

    ft = cfg["finetune"]
    full_epochs = int(ft.get("epochs",100))
    ramp_epochs = int(ft.get("ramp_epochs",30))
    epochs = 1 if args.smoke else (
        int(args.diagnostic_epochs) if args.diagnostic_epochs is not None else full_epochs
    )

    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(ft.get("lr",0.005)),
        momentum=float(ft.get("momentum",0.9)),
        weight_decay=float(ft.get("weight_decay",5e-4)),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(1,full_epochs))

    out_root = Path(cfg["output"]["root"]); out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / f"{cfg['experiment']['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if args.smoke:
        run_dir = Path(str(run_dir)+"_smoke")
    run_dir.mkdir(parents=True, exist_ok=False)

    save_json(cfg, run_dir/"config.json")
    save_json({
        "source_r4_checkpoint": args.source_r4_checkpoint,
        "source_r4_best_acc": src.get("best_acc"),
        "source_r4_best_epoch": src.get("best_epoch"),
        "lambda0_a8_fpstem_acc": acc_l0,
        "lambda1_scaled_w1a8_zero_epoch_acc": acc_l1,
        "initial_stem_stats": stem_stats(model),
    }, run_dir/"conversion.json")

    best_acc = acc_l1
    best_epoch = 0
    best_path = run_dir/"best.pt"
    last_path = run_dir/"last.pt"
    history = []

    model.conv1.set_progress_lambda(1.0)
    torch.save({
        "epoch":0,"model":model.state_dict(),"best_acc":best_acc,"best_epoch":0,
        "selection_lambda":1.0,"config":cfg,"source_r4_checkpoint":args.source_r4_checkpoint
    }, best_path)

    t0 = time.time()

    for epoch in range(epochs):
        lam = progressive_lambda(epoch, ramp_epochs)
        model.conv1.set_progress_lambda(lam)
        model.train()
        set_tau_all(model,tau)

        total = correct = 0
        loss_sum = 0.0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device)
            opt.zero_grad(set_to_none=True)
            logits=model(x)
            loss=criterion(logits,y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch+1}")
            loss.backward(); opt.step()
            loss_sum += loss.item()*y.size(0)
            correct += (logits.argmax(1)==y).sum().item()
            total += y.size(0)

        sched.step()

        _, scheduled_acc = evaluate(model,test_loader,criterion,device,force_lambda=lam)
        deploy_loss, deploy_acc = evaluate(model,test_loader,criterion,device,force_lambda=1.0)

        row = {
            "epoch":epoch+1,
            "lambda":lam,
            "lr":opt.param_groups[0]["lr"],
            "train_loss":loss_sum/max(total,1),
            "train_acc":100.0*correct/max(total,1),
            "scheduled_test_acc":scheduled_acc,
            "binary_lambda1_test_loss":deploy_loss,
            "binary_lambda1_test_acc":deploy_acc,
        }
        history.append(row)
        save_json(history,run_dir/"history.json")

        if deploy_acc > best_acc:
            best_acc = deploy_acc
            best_epoch = epoch+1
            old = model.conv1.progress_lambda
            model.conv1.set_progress_lambda(1.0)
            torch.save({
                "epoch":epoch+1,"model":model.state_dict(),"best_acc":best_acc,
                "best_epoch":best_epoch,"selection_lambda":1.0,
                "config":cfg,"source_r4_checkpoint":args.source_r4_checkpoint
            }, best_path)
            model.conv1.set_progress_lambda(old)

        torch.save({
            "epoch":epoch+1,"model":model.state_dict(),"best_acc":best_acc,
            "best_epoch":best_epoch,"training_lambda":lam,
            "config":cfg,"source_r4_checkpoint":args.source_r4_checkpoint
        }, last_path)

        print(f"epoch={epoch+1}/{epochs} lambda={lam:.4f} scheduled={scheduled_acc:.2f}% deploy_w1a8={deploy_acc:.2f}% best={best_acc:.2f}%")

    elapsed = time.time()-t0

    state = torch.load(best_path,map_location=device)
    model.load_state_dict(state["model"],strict=True)
    model.conv1.set_progress_lambda(1.0)
    _, reload_acc = evaluate(model,test_loader,criterion,device,force_lambda=1.0)

    k,b = folded_stem_bn_params(model)
    summary = {
        "source_r4_best_acc":src.get("best_acc"),
        "source_r4_best_epoch":src.get("best_epoch"),
        "lambda0_a8_fpstem_acc":acc_l0,
        "lambda1_scaled_w1a8_zero_epoch_acc":acc_l1,
        "best_deploy_w1a8_acc":best_acc,
        "best_epoch":best_epoch,
        "reload_deploy_w1a8_acc":reload_acc,
        "final_epoch_deploy_w1a8_acc":history[-1]["binary_lambda1_test_acc"] if history else acc_l1,
        "final_lambda":model.conv1.progress_lambda,
        "elapsed_s":elapsed,
        "trainable_params":sum(p.numel() for p in model.parameters() if p.requires_grad),
        "final_stem_stats":stem_stats(model),
        "folded_stem_bn_stats":{
            "k_min":float(k.min()),"k_max":float(k.max()),
            "k_abs_mean":float(k.abs().mean()),"bias_min":float(b.min()),"bias_max":float(b.max())
        },
        "git_commit":git_commit(),
    }
    save_json(summary,run_dir/"summary.json")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
