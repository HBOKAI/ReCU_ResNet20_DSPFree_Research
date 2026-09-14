"""Build mechanical tables and a verified file manifest for the 2026-09-14 archive.

Narrative conclusions live in the Markdown documents. This helper only
transcribes audited numbers, frozen width maps, and copied-file provenance.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "EXPERIMENT_SUMMARY_20260914"
TABLES = ARCHIVE / "tables"


def read_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


H0 = read_json("H0_INTEGER_BIAS_SWEEP_RESULTS.json")
H1 = read_json("H1_INTEGER_ACTIVATION_SWEEP_RESULTS.json")
H1S = read_json("H1S_SINGLE_NODE_RESULTS.json")
H1MP = read_json("H1MP_SEARCH_RESULTS.json")
H2A = read_json("H2_FINITE_WIDTH_RESULTS.json")
H2B = read_json("H2B_WIDTH_OPTIMIZATION_RESULTS.json")


COLUMNS = [
    "Experiment", "Source", "Main change", "Best/Test Accuracy",
    "Delta vs Source", "Params", "Binary Ops / MAC-equivalent",
    "Bias precision", "Activation precision", "Accumulator precision",
    "Training/Fine-tune", "Status", "Decision", "Evaluation split", "Evidence",
]


def row(
    experiment: str, source: str, change: str, accuracy: float | None,
    source_accuracy: float | None, params: int | None, ops: str,
    bias: str, activation: str, accumulator: str, training: str,
    status: str, decision: str, evidence: str,
    split: str = "CIFAR-10 official TEST",
) -> dict:
    delta = None if accuracy is None or source_accuracy is None else round(accuracy - source_accuracy, 2)
    return dict(zip(COLUMNS, [
        experiment, source, change, accuracy, delta, params, ops,
        bias, activation, accumulator, training, status, decision, split, evidence,
    ]))


RECU_OPS = "40,108,032 binary-conv terms; 442,368 FP stem products; 640 FP FC products"
R5_OPS = "40,108,032 binary-conv terms; 442,368 W1A8 select/negate stem terms; 640 FP FC products"
THERMO_OPS = "54,263,808 binary-conv terms; 640 classifier terms"
R8B_OPS = "54,263,808 binary-conv terms; 640 signed-pow2 FC shift terms"


def build_experiments() -> list[dict]:
    e = []
    add = e.append
    add(row("FP32 ResNet20", "prior baseline", "full precision reference", 91.06, None, None,
            "NOT FOUND", "FP32", "FP32", "FP32", "prior run", "HISTORICAL",
            "context only; no current workspace result artifact", "PROMPT_CODEX_R1_R2_R3.md", "reported TEST; artifact NOT FOUND"))
    add(row("Minimal W1A1", "prior baseline", "minimal binary model", 72.60, None, None,
            "NOT FOUND", "unknown", "W1A1", "unknown", "prior run", "HISTORICAL",
            "accuracy too low for selected line", "PROMPT_CODEX_R1_R2_R3.md", "reported TEST; artifact NOT FOUND"))
    add(row("Early ReCU failed adaptation", "prior adaptation", "non-official ReCU adaptation", 43.29, None, None,
            "NOT FOUND", "unknown", "unknown", "unknown", "prior run", "HISTORICAL",
            "exclude from official reproduction comparison", "PROMPT_CODEX_R1_R2_R3.md", "reported TEST; artifact NOT FOUND"))
    add(row("Early SiMaN failed adaptation", "prior adaptation", "non-official SiMaN adaptation", None, None, None,
            "NOT FOUND", "unknown", "unknown", "unknown", "prior run", "HISTORICAL",
            "numeric result NOT FOUND", "PROMPT_CODEX_R1_R2_R3.md", "NOT FOUND"))
    add(row("Official ReCU", "scratch", "official ReCU 600-epoch reproduction", 87.28, None, 270858,
            RECU_OPS, "FP BN/FC", "W1A1 backbone; FP stem", "not finite-width", "600 epochs from scratch", "SELECTED",
            "reproduced official architecture; source for R1–R3", "R1_R2_R3_FINAL_REPORT.md; experiments/recu_runs/...234802/history.json"))
    add(row("H1 QRPReLU scratch", "scratch", "PReLU to QRPReLU", 84.28, None, 271530,
            RECU_OPS, "FP", "W1A1 backbone; FP stem", "not finite-width", "600 epochs from scratch", "ABLATION",
            "inferior to warm-start R2", "experiments/recu_runs/...235444/history.json"))
    add(row("H2 QRPReLU + pow2 alpha scratch", "scratch", "QRPReLU and pow2 alpha", 84.60, None, 271530,
            RECU_OPS, "FP", "W1A1 backbone; FP stem", "not finite-width", "600 epochs from scratch", "ABLATION",
            "historical hardware-aware scratch result", "experiments/recu_runs/...235445/history.json"))
    add(row("R1 Remove alpha", "Official ReCU", "alpha=1", 87.02, 87.28, 270186,
            RECU_OPS, "FP", "W1A1 backbone; FP stem", "not finite-width", "100-epoch warm-start", "ABLATION",
            "recoverable after fine-tuning; no zero-shot conversion", "R1_R2_R3_FINAL_REPORT.md; r1_summary.json"))
    add(row("R2 Warm-start QRPReLU", "Official ReCU", "warm-start PReLU to QRPReLU", 86.72, 87.28, 271530,
            RECU_OPS, "FP", "W1A1 backbone; FP stem", "not finite-width", "100-epoch warm-start", "SELECTED",
            "QRPReLU source for R4", "R1_R2_R3_FINAL_REPORT.md; r2_summary.json"))
    add(row("R3 Fused pow2 affine", "Official ReCU", "fold alpha+BN and quantize K", 85.50, 87.28, 270186,
            RECU_OPS, "FP", "W1A1 backbone; FP stem", "not finite-width", "100-epoch warm-start", "ABLATION",
            "validated float fold and signed-pow2 branch", "R1_R2_R3_FINAL_REPORT.md; r3_summary.json"))
    add(row("R4 Multiplier-free backbone", "R2", "fused signed-pow2 alpha/BN", 86.11, 86.72, 270858,
            RECU_OPS, "FP", "W1A1 backbone; FP stem", "not finite-width", "100-epoch warm-start", "SELECTED",
            "hardware-friendly backbone source", "R4_FINAL_REPORT.md; r4_summary.json"))
    add(row("R5 Naive W1A8 stem", "R4", "3-channel W1A8 stem", 83.77, 86.11, 270858,
            R5_OPS, "FP", "W1A8 stem; W1A1 backbone", "not finite-width", "100-epoch fine-tune", "REJECTED",
            "below 84.5% go/no-go", "R5_FINAL_REPORT.md; r5_summary.json"))
    add(row("R5AB Progressive scaled W1A8", "R4", "scaled binary stem; progressive lambda", 83.96, 86.11, 270858,
            R5_OPS, "FP", "W1A8 stem; W1A1 backbone", "not finite-width", "100-epoch fine-tune", "REJECTED",
            "+0.19 pp vs R5 but below threshold", "R5AB_FINAL_REPORT.md; r5ab_summary.json"))
    add(row("R5T T1 FP stem", "R4", "Thermometer R=8; FP latent stem", 85.29, 86.11, 284250,
            "40,108,032 binary backbone terms; 14,155,776 FP stem product sites", "FP", "A1 thermometer; FP stem", "not finite-width", "100-epoch adaptation", "ABLATION",
            "initialization source for R5T T2", "R5T_FINAL_REPORT.md; r5t_t1_summary.json"))
    add(row("R5T T2 W1A1 stem", "R5T T1", "binary thermometer stem", 85.00, 85.29, 284250,
            THERMO_OPS, "FP", "W1A1 stem/backbone", "not finite-width", "100-epoch fine-tune", "SELECTED",
            "binary stem exceeds R5/R5AB", "R5T_FINAL_REPORT.md; r5t_t2_summary.json"))
    add(row("R5T-Long T1 FP stem", "R4", "200-epoch thermometer adaptation", 85.89, 86.11, 284250,
            "40,108,032 binary backbone terms; 14,155,776 FP stem product sites", "FP", "A1 thermometer; FP stem", "not finite-width", "200 epochs", "ABLATION",
            "+0.60 pp vs old T1", "R5T_LONG_FINAL_REPORT.md; r5t_long_t1_summary.json"))
    add(row("R5T-Long T2 W1A1 stem", "R5T-Long T1", "200-epoch binary-stem adaptation", 85.14, 85.89, 284250,
            THERMO_OPS, "FP", "W1A1 stem/backbone", "not finite-width", "200 epochs", "SELECTED",
            "source for R6", "R5T_LONG_FINAL_REPORT.md; r5t_long_t2_summary.json"))
    add(row("R6 Stem BN to pow2 affine", "R5T-Long T2", "remove stem BN; signed-pow2 K", 85.52, 85.14, 284250,
            THERMO_OPS, "FP B", "W1A1 stem/backbone", "not finite-width", "100-epoch fine-tune", "SELECTED",
            "exact fold and fine-tune gain", "R6_FINAL_REPORT.md; r6_summary.json"))
    add(row("R7 Head BN to pow2 affine", "R6", "remove head BN; signed-pow2 K", 85.22, 85.52, 284250,
            THERMO_OPS, "FP B", "W1A1 stem/backbone", "not finite-width", "100-epoch fine-tune", "SELECTED",
            "head path without standalone BN", "R7_FINAL_REPORT.md; r7_summary.json"))
    add(row("R8A W1 FC", "R7", "binary FC weights", 82.84, 85.22, 284250,
            "54,263,808 binary-conv terms; 640 sign-select FC terms", "FP B/FC bias", "W1A1 stem/backbone", "not finite-width", "100-epoch fine-tune", "REJECTED",
            "2.56 pp below R8B", "R8_FINAL_REPORT.md; r8a_summary.json"))
    add(row("R8B Signed-pow2 FC", "R7", "signed-pow2 FC weights", 85.40, 85.22, 284250,
            R8B_OPS, "FP B/FC bias", "W1A1 stem/backbone", "not finite-width", "100-epoch fine-tune", "SELECTED",
            "formal software model baseline", "R8_FINAL_REPORT.md; r8b_summary.json"))
    for item in H0["results"]:
        bits = int(item["bits"])
        add(row(f"H0 INT{bits} bias", "R8B", f"quantize 21 bias tensors to INT{bits}", float(item["accuracy"]), 85.40, 284250,
                R8B_OPS, f"INT{bits}", "float/reference", "not finite-width", "no training", "SELECTED" if bits == 6 else "ABLATION",
                "production bias setting" if bits == 6 else "comparison point", "H0_INTEGER_BIAS_SWEEP_RESULTS.json"))
    for item in H1["results"]:
        bits = int(item["bits"])
        add(row(f"H1 uniform INT{bits}", "H0 INT6", f"uniform activation/residual INT{bits}", float(item["accuracy"]), 85.13, 284250,
                R8B_OPS, "INT6", f"uniform INT{bits}", "not fully finite-width", "no training", "FAIL",
                "uniform precision rejected", "H1_INTEGER_ACTIVATION_SWEEP_RESULTS.json"))
    add(row("H1-S single-node sensitivity", "H0 INT6", "58 single-node INT8 tests and four scale policies", None, None, 284250,
            R8B_OPS, "INT6", "one INT8 node at a time", "not fully finite-width", "no training", "ABLATION",
            "rank sensitive nodes for mixed precision", "H1S_SINGLE_NODE_RESULTS.json", "TRAIN search-validation only; no final official TEST"))
    add(row("H1-MP mixed precision", "H0 INT6", "58-node mixed bits/shifts/policies", float(H1MP["final_plan"]["official_test_accuracy"]), 85.13, 284250,
            R8B_OPS, "INT6", "58-node mixed; mean 6.6897 bits", "not fully finite-width", "no training", "SELECTED",
            "official 85.03%; 18/18 residual alignments", "H1MP_SEARCH_RESULTS.json"))
    add(row("H2A original", "H1-MP", "finite-width anchor with rounded GAP >>6", 84.92, 85.03, 284250,
            R8B_OPS, "INT6", "H1MP frozen", "exact/conservative; GAP rounded", "no training", "FAIL",
            "-0.11 pp; original result artifact NOT FOUND", "conversation execution record; original JSON/report NOT FOUND", "reported official TEST; original artifact NOT FOUND"))
    add(row("H2A-v2 safe profile", "H1-MP", "GAP q_sum with scale shift +6", float(H2A["h2a_exact_width_anchor"]["accuracy"]), 85.03, 284250,
            R8B_OPS, "INT6", "H1MP frozen", "GAP16; FC24; logits24; QRP inner30/out38", "no training", "SAFE PROFILE",
            "zero-loss finite-width profile", "H2_FINITE_WIDTH_RESULTS.json"))
    add(row("H2B aggressive profile", "H2A-v2", "TRAIN-only serial internal-width reduction", float(H2B["final_official_test"]["accuracy"]), 85.03, 284250,
            R8B_OPS, "INT6", "H1MP frozen", "GAP13; FC16; logits16; QRP inner27/out35", "no training", "AGGRESSIVE PROFILE",
            "optional minimum-width profile; keep safe profile", "H2B_WIDTH_OPTIMIZATION_RESULTS.json"))
    return e


def write_csv(path: Path, rows: list[dict], columns: list[str]):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def build_progression() -> list[dict]:
    stages = [
        ("Official ReCU", 87.28, True, "official reproduced reference"),
        ("R1", 87.02, False, "alpha removal needs fine-tune; ablation"),
        ("R2", 86.72, True, "warm QRPReLU source for R4"),
        ("R3", 85.50, False, "separate affine ablation"),
        ("R4", 86.11, True, "multiplier-free backbone"),
        ("R5", 83.77, False, "3-channel W1A8 below threshold"),
        ("R5AB", 83.96, False, "small recovery; below threshold"),
        ("R5T", 85.00, True, "thermometer W1A1 stem"),
        ("R5T-Long", 85.14, True, "source for R6"),
        ("R6", 85.52, True, "stem BN removed"),
        ("R7", 85.22, True, "head BN removed"),
        ("R8A", 82.84, False, "binary FC accuracy loss"),
        ("R8B", 85.40, True, "selected software model"),
        ("H0 INT6", 85.13, True, "production integer bias"),
        ("H1 uniform INT8", 77.07, False, "uniform precision failed"),
        ("H1-MP", 85.03, True, "58-node mixed precision"),
        ("H2A", 84.92, False, "rounded GAP caused >0.10 pp loss; source artifact missing"),
        ("H2A-v2", 85.03, True, "safe zero-loss profile"),
        ("H2B", 84.93, False, "aggressive alternative; does not replace safe"),
    ]
    rows = []
    selected_accuracy = None
    for stage, accuracy, selected, reason in stages:
        rows.append({
            "stage": stage,
            "accuracy": accuracy,
            "delta_from_previous_selected": None if selected_accuracy is None else round(accuracy - selected_accuracy, 2),
            "delta_from_R8B": round(accuracy - 85.40, 2),
            "selected_or_not": "selected" if selected else "not selected",
            "reason": reason,
        })
        if selected:
            selected_accuracy = accuracy
    return rows


def build_widths() -> list[dict]:
    rows = []
    h1_plan = H1MP["final_plan"]
    safe = H2A["h2a_plan"]
    aggressive = H2B["h2b_search"]["final_plan"]
    for profile, plan in (("H2A-v2 SAFE", safe), ("H2B AGGRESSIVE", aggressive)):
        for node in H1MP["quant_nodes"]:
            name = node if isinstance(node, str) else node.get("name") or node.get("node")
            rows.append({
                "profile": profile,
                "group": "H1MP activation/residual",
                "node": name,
                "bits": int(h1_plan["bits_by_node"][name]),
                "shift": int(h1_plan["shift_by_node"][name]),
                "policy": str(h1_plan["policy_by_node"][name]),
                "semantics": "frozen mixed-precision integer boundary",
                "source": "H1MP_SEARCH_RESULTS.json",
            })
        for name, item in plan["binary_convolution"].items():
            rows.append({"profile": profile, "group": "binary accumulator", "node": name,
                         "bits": int(item["signed_acc_bits"]), "shift": "", "policy": "exact",
                         "semantics": "XNOR/popcount signed accumulator", "source": "H2_FINITE_WIDTH_RESULTS.json"})
        for name, item in plan["qrprelu"].items():
            for part, key in (("inner", "inner_bits"), ("output", "bits")):
                rows.append({"profile": profile, "group": f"QRPReLU {part}", "node": name,
                             "bits": int(item[key]), "shift": int(item["param_shift"] if part == "inner" else item["output_shift"]),
                             "policy": item["mode"], "semantics": "internal finite integer",
                             "source": "H2_FINITE_WIDTH_RESULTS.json" if profile.startswith("H2A") else "H2B_WIDTH_OPTIMIZATION_RESULTS.json"})
        for group, node, bits, shift, semantics in (
            ("GAP", "gap.sum", plan["gap"]["sum_bits"], plan["gap"]["input_shift"], "q_gap=q_sum; output shift=input shift+6"),
            ("GAP", "gap.output", plan["gap"]["output_bits"], plan["gap"]["output_shift"], "deferred scale; no /64 rounding"),
            ("FC", "fc.accumulator", plan["fc"]["accumulator_bits"], plan["fc"]["common_shift"], "signed-pow2 shifted-product accumulation"),
            ("FC", "fc.bias_add", plan["fc"]["bias_add_bits"], plan["fc"]["bias_add_shift"], "H0 INT6 bias add"),
            ("FC", "final_logits", plan["fc"]["final_logits_bits"], plan["fc"]["bias_add_shift"], "finite output logits"),
            ("bias", "H0 all 21 bias tensors", 6, "per-tensor", "762 integer q values plus power-of-two scale"),
        ):
            rows.append({"profile": profile, "group": group, "node": node, "bits": int(bits),
                         "shift": shift, "policy": "frozen" if group == "bias" else "finite",
                         "semantics": semantics,
                         "source": "H0_INTEGER_BIAS_SWEEP_RESULTS.json" if group == "bias" else
                         ("H2_FINITE_WIDTH_RESULTS.json" if profile.startswith("H2A") else "H2B_WIDTH_OPTIMIZATION_RESULTS.json")})
    return rows


ALIASED_JSON = {
    "official_recu_history.json": "experiments/recu_runs/recu_resnet20_official_repro_s123_20260911_234802/history.json",
    "h1_qrprelu_scratch_history.json": "experiments/recu_runs/recu_resnet20_qrprelu_s123_20260911_235444/history.json",
    "h2_qrprelu_pow2alpha_scratch_history.json": "experiments/recu_runs/recu_resnet20_qrprelu_pow2alpha_s123_20260911_235445/history.json",
    "r1_summary.json": "experiments/recu_r1_remove_alpha/recu_r1_remove_alpha_20260912_094409/summary.json",
    "r2_summary.json": "experiments/recu_r2_warmstart_qrprelu/recu_r2_warmstart_qrprelu_20260912_094409/summary.json",
    "r3_summary.json": "experiments/recu_r3_fused_pow2_affine/recu_r3_fused_pow2_affine_20260912_094409/summary.json",
    "r4_summary.json": "experiments/recu_r4/recu_r4_qrprelu_fused_pow2_affine_20260912_110737/summary.json",
    "r5_summary.json": "experiments/recu_r5_w1a8_stem/recu_r5_w1a8_stem_20260912_122359/summary.json",
    "r5ab_summary.json": "experiments/recu_r5ab/recu_r5ab_progressive_scaled_w1a8_20260912_132942/summary.json",
    "r5t_t1_summary.json": "experiments/recu_r5t/recu_r5t_thermometer_r8_20260912_143431/summary.json",
    "r5t_t2_summary.json": "experiments/recu_r5t/recu_r5t_thermometer_r8_20260912_150935/summary.json",
    "r5t_long_t1_summary.json": "experiments/recu_r5t_long/recu_r5t_thermometer_r8_long_20260912_155005/summary.json",
    "r5t_long_t2_summary.json": "experiments/recu_r5t_long/recu_r5t_thermometer_r8_long_20260912_171239/summary.json",
    "r6_summary.json": "experiments/recu_r6/recu_r6_r5tlong_stem_bn_pow2_20260912_203911/summary.json",
    "r7_summary.json": "experiments/recu_r7/recu_r7_r6_head_bn_pow2_20260912_220758/summary.json",
    "r8a_summary.json": "experiments/recu_r8a/recu_r8a_r7_w1_fc_20260912_225827/summary.json",
    "r8b_summary.json": "experiments/recu_r8b/recu_r8b_r7_pow2_fc_20260912_225827/summary.json",
}


def file_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def stage_of(name: str) -> str:
    name = name.upper()
    if name.startswith("H2_FINITE_WIDTH"):
        return "H2A-v2"
    if name.startswith("RECU_OFFICIAL"):
        return "Official ReCU"
    if name.startswith("RECU_R5T_THERMOMETER_R8_LONG"):
        return "R5T-Long"
    if name.startswith("RECU_R4"):
        return "R4"
    if name.startswith("RECU_R6"):
        return "R6"
    if name.startswith("RECU_R7"):
        return "R7"
    if name.startswith("RECU_R8B"):
        return "R8B"
    if name.startswith("R1_R2_R3") or name.startswith(("R1_", "R2_", "R3_")):
        return "R1/R2/R3"
    if name.startswith("R5T_LONG"):
        return "R5T-Long"
    if name.startswith("R5T"):
        return "R5T"
    if name.startswith("R5AB"):
        return "R5AB"
    if name.startswith("R8"):
        return "R8A/R8B"
    for prefix in ("R4", "R5", "R6", "R7", "H0", "H1S", "H1MP", "H1", "H2B", "H2"):
        if name.startswith(prefix):
            return prefix
    if name.startswith("OFFICIAL_RECU"):
        return "Official ReCU"
    if name.startswith("H1_QRPRELU_SCRATCH"):
        return "H1 scratch"
    if name.startswith("H2_QRPRELU_POW2ALPHA"):
        return "H2 scratch"
    return "source audit / final pipeline"


NOT_FOUND = [
    ("FP32 ResNet20 baseline result/checkpoint", "only prior prompt 91.06%; no current workspace artifact"),
    ("Minimal W1A1 baseline result/checkpoint", "only prior prompt 72.60%; no current workspace artifact"),
    ("Early ReCU failed adaptation result/checkpoint", "only prior prompt 43.29%; no current workspace artifact"),
    ("Early SiMaN failed adaptation result/checkpoint", "no numeric result or artifact in current workspace"),
    ("Original H2A pre-v2 result JSON/report", "root H2 report/result now contain H2A-v2; original 84.92% exists only in conversation execution record"),
]


def build_manifest() -> str:
    lines = [
        "# File Manifest",
        "",
        "The archive contains copied small reports, result JSON, run summaries/histories, final-lineage configs, and generated summary files/tables. Checkpoints, datasets, caches, and full experiment directories were not copied. Original paths are relative to the workspace unless marked absolute. SHA256 of each copied file is checked against its source. Timestamps are UTC modification times. This manifest excludes its own hash to avoid a circular hash definition.",
        "",
        "| Archive relative path | Original workspace path | Type | Stage | Bytes | Modified UTC | SHA256 |",
        "|---|---|---|---|---:|---|---|",
    ]
    counts = {"reports": 0, "results_json": 0, "configs": 0}
    for category in ("reports", "results_json", "configs"):
        for copied in sorted((ARCHIVE / category).iterdir(), key=lambda p: p.name.lower()):
            if not copied.is_file():
                continue
            source_rel = ALIASED_JSON.get(copied.name, copied.name)
            if category == "configs":
                source_rel = f"configs/{copied.name}"
            source = ROOT / source_rel
            if not source.is_file():
                raise FileNotFoundError(f"Archive source missing: {source}")
            source_hash = file_sha256(source)
            archive_hash = file_sha256(copied)
            if source_hash != archive_hash:
                raise RuntimeError(f"Archive copy hash mismatch: {copied}")
            stamp = datetime.fromtimestamp(source.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            rel = copied.relative_to(ARCHIVE).as_posix()
            lines.append(
                f"| `{rel}` | `{source_rel}` | {copied.suffix.lstrip('.').upper()} | {stage_of(copied.name)} | "
                f"{copied.stat().st_size} | {stamp} | `{archive_hash}` |"
            )
            counts[category] += 1
    for generated in sorted(ARCHIVE.glob("*.md")) + sorted(TABLES.glob("*")):
        if generated.name == "08_FILE_MANIFEST.md" or not generated.is_file():
            continue
        stamp = datetime.fromtimestamp(generated.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        rel = generated.relative_to(ARCHIVE).as_posix()
        lines.append(
            f"| `{rel}` | archive-generated from sources listed above | {generated.suffix.lstrip('.').upper()} | "
            f"summary / table | {generated.stat().st_size} | {stamp} | `{file_sha256(generated)}` |"
        )
    lines += [
        "",
        f"Copied-file totals: **{counts['reports']} reports**, **{counts['results_json']} JSON**, **{counts['configs']} configs**.",
        "",
        "## Expected historical sources not found",
        "",
        "| Expected source | Status | Evidence / limitation |",
        "|---|---|---|",
    ]
    for label, note in NOT_FOUND:
        lines.append(f"| {label} | **NOT FOUND** | {note} |")
    lines += [
        "",
        "The H2A-v2 file is not a substitute for the original H2A file. Historical scores with missing source artifacts are labeled as reported figures, not as independently reproducible workspace records.",
        "",
    ]
    return "\n".join(lines)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    experiments = build_experiments()
    write_csv(TABLES / "experiment_results.csv", experiments, COLUMNS)
    (TABLES / "experiment_results.json").write_text(
        json.dumps({"schema": COLUMNS, "rows": experiments}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    progression = build_progression()
    write_csv(TABLES / "accuracy_progression.csv", progression, list(progression[0].keys()))
    widths = build_widths()
    write_csv(TABLES / "hardware_widths.csv", widths, list(widths[0].keys()))
    (ARCHIVE / "08_FILE_MANIFEST.md").write_text(build_manifest(), encoding="utf-8")
    verify_archive()
    print(f"rows={len(experiments)} progression={len(progression)} width_rows={len(widths)} archive_verified=True")


def verify_archive():
    assert len(list(ARCHIVE.glob("*.md"))) == 10
    for name, expected_columns, expected_rows in (
        ("experiment_results.csv", COLUMNS, 36),
        ("accuracy_progression.csv", list(build_progression()[0].keys()), 19),
        ("hardware_widths.csv", list(build_widths()[0].keys()), 172),
    ):
        with (TABLES / name).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            assert reader.fieldnames == expected_columns, name
            rows = list(reader)
            assert len(rows) == expected_rows and all(None not in row for row in rows), name
    result = json.loads((TABLES / "experiment_results.json").read_text(encoding="utf-8"))
    assert result["schema"] == COLUMNS and len(result["rows"]) == 36
    assert [r["Experiment"] for r in result["rows"]] == [r["Experiment"] for r in build_experiments()]
    for path in ARCHIVE.rglob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))
    for document in ARCHIVE.glob("*.md"):
        content = document.read_text(encoding="utf-8")
        assert content.startswith("# ") and len(content) > 100, document
        for link in re.findall(r"\]\(([^)]+)\)", content):
            if "://" not in link:
                assert (document.parent / link).exists(), (document, link)
    manifest = (ARCHIVE / "08_FILE_MANIFEST.md").read_text(encoding="utf-8")
    entries = {}
    for line in manifest.splitlines():
        match = re.match(r"\| `([^`]+)` \| .* \| `([0-9a-f]{64})` \|$", line)
        if match:
            entries[match.group(1)] = match.group(2)
    files = [p for p in ARCHIVE.rglob("*") if p.is_file()]
    assert len(entries) == len(files) - 1
    for file in files:
        assert file.suffix.lower() not in {".pt", ".pth"}
        rel = file.relative_to(ARCHIVE).as_posix()
        if rel != "08_FILE_MANIFEST.md":
            assert entries[rel] == file_sha256(file), rel
    stages = {row["stage"] for row in build_progression()}
    assert {"Official ReCU", "R1", "R2", "R3", "R4", "R5", "R5AB", "R5T", "R5T-Long", "R6", "R7", "R8A", "R8B", "H0 INT6", "H1 uniform INT8", "H1-MP", "H2A", "H2A-v2", "H2B"} <= stages


if __name__ == "__main__":
    main()
