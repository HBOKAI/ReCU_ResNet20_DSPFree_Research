"""Integrity checks for the completed R8B-LONG control artifacts."""

import csv
import hashlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "r8b_long" / "R8B_LONG_RESULTS.json"
RUN = ROOT / "experiments" / "recu_r8b_long" / "r8b_long_s123_20260916_005803"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_results_cover_full_run_and_baseline():
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    assert len(data["history"]) == 600
    assert [m["epoch"] for m in data["milestones"]] == [100, 200, 300, 400, 500, 600]
    assert data["baseline_reproduction"]["pass"] is True
    assert data["baseline_reproduction"]["accuracy"] == 85.40
    assert data["best_official_test_accuracy"] == 84.91
    assert abs(data["best_official_test_delta_pp"] - (-0.49)) < 1e-9


def test_milestone_checkpoint_hashes_and_epochs():
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    for item in data["milestones"]:
        path = RUN / f"epoch_{item['epoch']:03d}.pt"
        assert path.exists()
        assert _sha(path) == item["checkpoint_sha256"]
        state = torch.load(path, map_location="cpu", weights_only=False)
        assert int(state["epoch"]) == item["epoch"]
    best = RUN / "best_validation.pt"
    assert _sha(best) == data["best_validation_checkpoint_sha256"]


def test_curve_split_and_recipe():
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    with (ROOT / "reports" / "r8b_long" / "R8B_LONG_CURVE.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 600
    assert rows[-1]["epoch"] == "600"
    assert sum(1 for r in rows if r["official_test_accuracy"]) == 6
    assert data["split"]["train_count_used"] == 45000
    assert data["split"]["validation_count_used"] == 5000
    recipe = (ROOT / "reports" / "r8b_long" / "R8B_LONG_RECIPE.md").read_text(encoding="utf-8")
    assert "CosineAnnealingLR(T_max=100)" in recipe
    assert "CosineAnnealingLR(T_max=600)" in recipe
