import csv
import json
import random
import subprocess
from datetime import datetime
from pathlib import Path

import torch


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def make_run_dir(config, smoke=False):
    root = Path(config["output"].get("root", "experiments/recu_runs"))
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = config["experiment"]["name"]
    seed = config["experiment"].get("seed", 123)
    suffix = "_smoke" if smoke else ""
    p = root / f"{name}_s{seed}_{stamp}{suffix}"
    p.mkdir(parents=True, exist_ok=False)
    return p


def append_registry(row, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "experiment_id", "timestamp", "git_commit", "seed",
        "activation_mode", "alpha_mode", "params",
        "epochs", "batch_size", "lr", "weight_decay",
        "tau_min", "tau_max", "best_test_acc", "best_epoch",
        "checkpoint_path", "config_path", "status",
    ]
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in keys})
