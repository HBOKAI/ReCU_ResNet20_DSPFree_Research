"""H1-S: single-node INT8 sensitivity and train-derived scale-policy search."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recu_hw.h1s_mp_quant import CalibrationSampleCollector, SelectiveIntegerFakeQuant
from recu_hw.h1_search_workspace import (
    build_h1_search_loaders,
    evaluate,
    load_r8b_h0_model,
    make_h1_adapter,
)
from tools.eval_integer_bias_sweep import discover_formal_r8b_checkpoint


@torch.no_grad()
def calibrate(model, adapter, loader, device, sample_limit):
    collector = CalibrationSampleCollector(sample_cap_per_node=65536, values_per_batch=2048)
    adapter.set_fake_quant(None)
    adapter.set_observer(collector)
    seen = 0
    model.eval()
    for x, _ in loader:
        if seen >= sample_limit:
            break
        remain = sample_limit - seen
        if x.size(0) > remain:
            x = x[:remain]
        adapter(x.to(device, non_blocking=True))
        seen += int(x.size(0))
    adapter.set_observer(None)
    if seen != sample_limit:
        raise RuntimeError(f"Calibration saw {seen} samples, expected {sample_limit}")
    missing = set(adapter.list_quant_nodes()) - set(collector.stats)
    if missing:
        raise RuntimeError("Calibration missed nodes: " + ", ".join(sorted(missing)))
    return collector, seen


def eval_single_node(model, adapter, loader, device, node, bits, shift):
    fake_quant = SelectiveIntegerFakeQuant({node: bits}, {node: shift})
    adapter.set_observer(None)
    adapter.set_fake_quant(fake_quant)
    accuracy = evaluate(model, loader, device)
    adapter.set_fake_quant(None)
    return {
        "accuracy": float(accuracy),
        "saturation": int(fake_quant.total_saturation()),
        "saturation_by_node": dict(fake_quant.saturation_by_node),
    }


def render_report(result):
    lines = [
        "# H1-S Single-Node INT8 Sensitivity Report",
        "",
        "> Exactly one verified H1 node was quantized at a time. All policy and ranking decisions use held-out CIFAR-10 TRAIN data only.",
        "",
        "## Source and methodology",
        "",
        f"- R8B checkpoint: `{result['source']['r8b_checkpoint']}`",
        f"- H0 INT6 bias JSON: `{result['source']['h0_results_json']}`",
        f"- H0 search-validation baseline: **{result['source']['search_validation_baseline_accuracy']:.2f}%**",
        f"- H0 official-test baseline (source verification only): **{result['source']['h0_test_baseline_accuracy']:.2f}%**",
        f"- Calibration: CIFAR-10 TRAIN subset, {result['data']['calibration_samples']} samples.",
        f"- Search-validation: a non-overlapping CIFAR-10 TRAIN subset, {result['data']['search_validation_samples']} samples, deterministic raw-domain ToTensor.",
        f"- Split seed: `{result['data']['split_seed']}`; official test used for search: **False**.",
        "- H0 INT6 bias remained fixed. The 58 verified H1 boundaries and 18 residual sites were reused.",
        "",
        "## Single-node ranking",
        "",
        "| Rank | Node | Selected policy | Shift | Only-this-node INT8 accuracy | Drop vs search baseline | Saturation | Class |",
        "|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(result["final_single_node_sensitivity"], 1):
        lines.append(
            f"| {rank} | `{row['node']}` | `{row['selected_policy']}` | {row['shift']} | "
            f"{row['search_accuracy']:.2f}% | {row['drop_vs_search_baseline_pp']:+.2f} pp | "
            f"{row['search_saturation']} | `{row['sensitivity_class']}` |"
        )

    lines += ["", "## Top-10 scale-policy ablation", ""]
    for node in result["top_policy_nodes"]:
        lines += [
            f"### `{node}`",
            "",
            "| Policy | Shift | Search-validation accuracy | Drop | Saturation |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in result["top_policy_ablation"][node]:
            lines.append(
                f"| `{row['policy']}` | {row['shift']} | {row['search_accuracy']:.2f}% | "
                f"{row['drop_vs_search_baseline_pp']:+.2f} pp | {row['search_saturation']} |"
            )
        lines.append("")

    top10 = result["final_single_node_sensitivity"][:10]
    robust = [row["node"] for row in result["final_single_node_sensitivity"] if row["drop_vs_search_baseline_pp"] <= 0.02]
    obvious = [row["node"] for row in result["final_single_node_sensitivity"] if row["drop_vs_search_baseline_pp"] > 0.30]
    l3 = next(row for row in result["final_single_node_sensitivity"] if row["node"] == "layer3.0.qrprelu_output")
    top10_text = ", ".join("`" + row["node"] + "`" for row in top10)
    robust_text = ", ".join("`" + node + "`" for node in robust) if robust else "none"
    obvious_text = ", ".join("`" + node + "`" for node in obvious) if obvious else "none"
    policy_improved = []
    for node, rows in result["top_policy_ablation"].items():
        mse = next(row for row in rows if row["policy"] == "mse")
        best = rows[0]
        if best["search_accuracy"] > mse["search_accuracy"] + 1e-9:
            policy_improved.append(node)

    lines += [
        "## Interpretation",
        "",
        f"- Most sensitive nodes: {top10_text}.",
        f"- `layer3.0.qrprelu_output` remains rank {next(i for i, row in enumerate(result['final_single_node_sensitivity'], 1) if row['node'] == 'layer3.0.qrprelu_output')} with drop {l3['drop_vs_search_baseline_pp']:+.2f} pp.",
        f"- Robust/near-unaffected nodes (drop <=0.02 pp): {robust_text}.",
        f"- Clearly affected nodes (drop >0.30 pp): {obvious_text}.",
        f"- Policy ablation improved validation accuracy over MSE for: {', '.join(f'`{node}`' for node in policy_improved) if policy_improved else 'none'}.",
        "- Official test was not used to select policies, rank nodes, or choose any mixed-precision width.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "h1s_h1mp.json"))
    parser.add_argument("--source-r8b-checkpoint", default=None)
    parser.add_argument("--h0-results-json", default=str(ROOT / "H0_INTEGER_BIAS_SWEEP_RESULTS.json"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--calibration-samples", type=int, default=None)
    parser.add_argument("--search-validation-samples", type=int, default=None)
    parser.add_argument("--topk-policy-ablation", type=int, default=10)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    split_cfg = config.get("data_split", {})
    seed = int(args.seed if args.seed is not None else split_cfg.get("split_seed", 20260913))
    calibration_samples = int(args.calibration_samples or split_cfg.get("calibration_samples", 10000))
    search_samples = int(args.search_validation_samples or split_cfg.get("search_validation_samples", 10000))

    if args.source_r8b_checkpoint:
        source_checkpoint = Path(args.source_r8b_checkpoint)
        if not source_checkpoint.is_absolute():
            source_checkpoint = ROOT / source_checkpoint
        source_summary_path = source_checkpoint.parent / "summary.json"
        source_summary = json.loads(source_summary_path.read_text(encoding="utf-8")) if source_summary_path.exists() else {}
        candidate_summaries = [str(source_summary_path)]
    else:
        source_meta, candidates = discover_formal_r8b_checkpoint(ROOT)
        source_checkpoint = source_meta["checkpoint_path"]
        source_summary_path = source_meta["summary_path"]
        source_summary = source_meta["summary"]
        candidate_summaries = [str(path) for path in candidates]

    h0_path = Path(args.h0_results_json)
    if not h0_path.is_absolute():
        h0_path = ROOT / h0_path
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    model = load_r8b_h0_model(source_checkpoint, h0_path, device)
    adapter = make_h1_adapter(model)
    calibration_loader, search_loader, test_loader = build_h1_search_loaders(
        seed, calibration_samples, search_samples
    )
    baseline_search_acc = evaluate(model, search_loader, device)
    baseline_test_acc = evaluate(model, test_loader, device)
    collector, seen = calibrate(model, adapter, calibration_loader, device, calibration_samples)
    nodes = adapter.list_quant_nodes()
    if len(nodes) != 58:
        raise RuntimeError(f"Expected 58 H1 nodes, found {len(nodes)}")

    pass_a = []
    for index, node in enumerate(nodes, 1):
        shift = collector.choose_shift(node, 8, "mse")
        measured = eval_single_node(model, adapter, search_loader, device, node, 8, shift)
        drop = baseline_search_acc - measured["accuracy"]
        pass_a.append({
            "node": node,
            "bits": 8,
            "policy": "mse",
            "shift": int(shift),
            "search_accuracy": measured["accuracy"],
            "drop_vs_search_baseline_pp": float(drop),
            "search_saturation": measured["saturation"],
        })
        print(f"H1-S [{index}/{len(nodes)}] {node}: {measured['accuracy']:.2f}% drop={drop:+.2f} pp", flush=True)

    pass_a.sort(key=lambda row: row["drop_vs_search_baseline_pp"], reverse=True)
    top_nodes = [row["node"] for row in pass_a[:args.topk_policy_ablation]]
    policies = config.get("h1s", {}).get("scale_policies", ["absmax", "mse", "p99.9", "p99.99"])
    policy_ablation = {}
    selected_policy = {row["node"]: "mse" for row in pass_a}
    for node in top_nodes:
        rows = []
        for policy in policies:
            shift = collector.choose_shift(node, 8, policy)
            measured = eval_single_node(model, adapter, search_loader, device, node, 8, shift)
            rows.append({
                "policy": policy,
                "shift": int(shift),
                "search_accuracy": measured["accuracy"],
                "drop_vs_search_baseline_pp": float(baseline_search_acc - measured["accuracy"]),
                "search_saturation": measured["saturation"],
            })
        rows.sort(key=lambda row: (-row["search_accuracy"], row["search_saturation"]))
        policy_ablation[node] = rows
        selected_policy[node] = rows[0]["policy"]

    pass_a_by_node = {row["node"]: row for row in pass_a}
    sensitivity = []
    for node in nodes:
        policy = selected_policy[node]
        row = pass_a_by_node[node] if policy == "mse" else next(item for item in policy_ablation[node] if item["policy"] == policy)
        drop = float(row["drop_vs_search_baseline_pp"])
        sensitivity.append({
            "node": node,
            "bits": 8,
            "selected_policy": policy,
            "shift": int(row["shift"]),
            "search_accuracy": float(row["search_accuracy"]),
            "drop_vs_search_baseline_pp": drop,
            "search_saturation": int(row["search_saturation"]),
            "sensitivity_class": (
                "robust" if drop <= 0.02 else "mild" if drop <= 0.10 else "sensitive" if drop <= 0.30 else "very_sensitive"
            ),
        })
    sensitivity.sort(key=lambda row: row["drop_vs_search_baseline_pp"], reverse=True)

    output_cfg = config.get("output", {})
    output_dir = Path(args.output_dir or output_cfg.get("root", "."))
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / output_cfg.get("h1s_results", "H1S_SINGLE_NODE_RESULTS.json")
    report_path = output_dir / output_cfg.get("h1s_report", "H1S_SINGLE_NODE_REPORT.md")
    result = {
        "status": "COMPLETE",
        "source": {
            "r8b_checkpoint": str(source_checkpoint),
            "source_summary": str(source_summary_path),
            "candidate_formal_summaries": candidate_summaries,
            "r8b_stored_best_acc": source_summary.get("best_acc"),
            "r8b_stored_best_epoch": source_summary.get("best_epoch"),
            "h0_results_json": str(h0_path),
            "h0_test_baseline_accuracy": float(baseline_test_acc),
            "search_validation_baseline_accuracy": float(baseline_search_acc),
        },
        "data": {
            "calibration_source": "CIFAR-10 TRAIN only",
            "calibration_samples": int(seen),
            "search_validation_source": "non-overlapping CIFAR-10 TRAIN subset",
            "search_validation_samples": int(search_samples),
            "split_seed": seed,
            "official_test_used_for_search": False,
        },
        "quant_node_count": len(nodes),
        "quant_nodes": adapter.quant_node_metadata(),
        "calibration_ranges": collector.export(),
        "pass_a_int8_mse": pass_a,
        "top_policy_nodes": top_nodes,
        "top_policy_ablation": policy_ablation,
        "selected_policy_by_node": selected_policy,
        "final_single_node_sensitivity": sensitivity,
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report_path.write_text(render_report(result), encoding="utf-8")
    print(f"Wrote {result_path}", flush=True)
    print(f"Wrote {report_path}", flush=True)
    print("Top 10 H1-S nodes:", flush=True)
    for row in sensitivity[:10]:
        print(f"{row['node']}: drop={row['drop_vs_search_baseline_pp']:+.2f} pp policy={row['selected_policy']}", flush=True)


if __name__ == "__main__":
    main()
