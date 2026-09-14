import argparse, json, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from recu_hw.data import build_recu_loaders
from recu_hw.r4 import R4ResNet20
from recu_hw.r5ab import build_r5ab_from_r4, stem_stats

@torch.no_grad()
def eval_acc(model,loader,device,lam):
    old=model.conv1.progress_lambda
    model.conv1.set_progress_lambda(lam); model.eval()
    total=correct=0
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        p=model(x).argmax(1)
        correct += (p==y).sum().item(); total += y.size(0)
    model.conv1.set_progress_lambda(old)
    return 100.0*correct/max(total,1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source-r4-checkpoint",required=True)
    ap.add_argument("--input-scale-exp",type=int,default=-5)
    ap.add_argument("--device",default=None)
    args=ap.parse_args()
    device=torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    state=torch.load(args.source_r4_checkpoint,map_location=device)
    r4=R4ResNet20(num_classes=10).to(device)
    r4.load_state_dict(state["model"],strict=True); r4.eval()
    m=build_r5ab_from_r4(r4,args.input_scale_exp).to(device)

    bw=torch.where(m.conv1.weight>=0,torch.ones_like(m.conv1.weight),-torch.ones_like(m.conv1.weight))
    unique=sorted(set(float(x) for x in torch.unique(bw).cpu().tolist()))

    cfg={"dataset":{"name":"cifar10","root":"./data","download":True,"protocol":"official_recu_train_test"},
         "training":{"batch_size":256,"test_batch_size":128,"num_workers":0,"windows_safe_smoke":True}}
    train_loader,test_loader=build_recu_loaders(cfg,smoke=False)

    total=satl=sath=0
    for i,(x,_) in enumerate(train_loader):
        if i>=16: break
        s=m.input_quant.stats(x.to(device))
        total += s["total"]; satl += s["sat_low"]; sath += s["sat_high"]

    report={
        "source_r4_best_acc":state.get("best_acc"),
        "source_r4_best_epoch":state.get("best_epoch"),
        "binary_weight_unique_values":unique,
        "lambda0_a8_fpstem_acc":eval_acc(m,test_loader,device,0.0),
        "lambda1_scaled_w1a8_zero_epoch_acc":eval_acc(m,test_loader,device,1.0),
        "input_scale_exp":args.input_scale_exp,
        "input_scale":2.0**args.input_scale_exp,
        "saturation_fraction":(satl+sath)/max(total,1),
        "stem_stats":stem_stats(m),
    }
    print(json.dumps(report,indent=2))
    if report["saturation_fraction"]>1e-4:
        raise SystemExit("FAIL: saturation > 1e-4")

if __name__=="__main__":
    main()
