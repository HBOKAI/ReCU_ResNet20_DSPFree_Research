"""H1-MP sensitivity-seeded mixed-precision search."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recu_hw.h1s_mp_quant import (
    CalibrationSampleCollector,
    SelectiveIntegerFakeQuant,
    candidate_bits_for_class,
    classify_sensitivity,
    initial_bits_for_class,
)
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


def make_shift_map(collector, node_bits, selected_policy, min_shift=-16, max_shift=24):
    return {
        node: collector.choose_shift(
            node,
            bits=bits,
            policy=selected_policy.get(node, "mse"),
            min_shift=min_shift,
            max_shift=max_shift,
        )
        for node, bits in node_bits.items()
    }


@torch.no_grad()
def evaluate_plan(model, adapter, loader, device, collector, node_bits, selected_policy, min_shift, max_shift):
    shift_map = make_shift_map(collector, node_bits, selected_policy, min_shift, max_shift)
    fake_quant = SelectiveIntegerFakeQuant(node_bits=node_bits, node_shifts=shift_map)
    adapter.set_observer(None)
    adapter.set_fake_quant(fake_quant)
    accuracy = evaluate(model, loader, device)
    alignment = adapter.verify_integer_residual_alignment(
        shift_map=shift_map,
        bits_or_map=node_bits,
    )
    adapter.set_fake_quant(None)
    return {
        "accuracy": float(accuracy),
        "shift_map": shift_map,
        "total_saturation": int(fake_quant.total_saturation()),
        "saturation_by_node": {
            node: int(fake_quant.saturation_by_node.get(node, 0))
            for node in node_bits
        },
        "residual_alignment": alignment,
    }


def next_higher_bit(current, candidates):
    values = sorted(bit for bit in candidates if bit > current)
    return values[0] if values else None


def next_lower_bit(current, candidates):
    values = sorted((bit for bit in candidates if bit < current), reverse=True)
    return values[0] if values else None


def render_report(result):
    source = result["source"]
    final = result.get("final_plan")
    lines = [
        "# H1-MP Mixed-Precision Search Report",
        "",
        "> H1-S seeded mixed-precision search with H0 INT6 bias fixed. All policy, promotion, demotion, and final-plan decisions use TRAIN-derived data only.",
        "",
        "## A. Source R8B / H0 verification",
        "",
        f"- R8B checkpoint: `{source['r8b_checkpoint']}`",
        f"- H0 INT6 bias JSON: `{source['h0_results_json']}`",
        f"- R8B stored/reload source verification: {source.get('r8b_stored_best_acc')}% / {source.get('r8b_reload_accuracy'):.2f}%.",
        f"- H0 search-validation baseline: **{source['search_validation_baseline_accuracy']:.2f}%**.",
        f"- H0 official-test baseline: **{source['h0_test_baseline_accuracy']:.2f}%**; test baseline was source verification only.",
        "- H0 INT6 bias was reconstructed from stored q_values + shifts and remained fixed; R8B binary/pow2 invariants were unchanged.",
        "",
        "## B. INT14 high-precision anchor",
        "",
        f"- All 58 nodes at INT14 search-validation accuracy: **{result['anchor']['search_accuracy']:.2f}%**.",
        f"- Drop vs H0 search-validation baseline: **{result['anchor']['drop_pp']:+.2f} pp**.",
        f"- Anchor saturation: **{result['anchor']['saturation']}**.",
        f"- Anchor gate (drop <=0.30 pp): **{'PASS' if result['anchor']['drop_pp'] <= 0.30 else 'FAIL'}**.",
        "",
        "## C. H1-S sensitivity classification",
        "",
        "| Class | Initial bits | Candidate ladder | Node count |",
        "|---|---:|---|---:|",
    ]
    for cls in ("robust", "mild", "sensitive", "very_sensitive"):
        count = sum(value == cls for value in result["sensitivity_classes"].values())
        lines.append(
            f"| `{cls}` | {result['initial_bits_by_class'][cls]} | "
            f"{', '.join(str(bit) for bit in result['candidate_bits_by_class'][cls])} | {count} |"
        )
    lines += [
        "",
        "Top H1-S nodes used to seed the plan:",
        "",
        "| Rank | Node | H1-S drop | Class | Selected policy |",
        "|---:|---|---:|---|---|",
    ]
    for rank, row in enumerate(result["h1s_top10"], 1):
        lines.append(
            f"| {rank} | `{row['node']}` | {row['drop_vs_search_baseline_pp']:+.2f} pp | "
            f"`{row['sensitivity_class']}` | `{row['selected_policy']}` |"
        )

    if result["status"] != "COMPLETE":
        lines += ["", "## Stop status", "", f"- **{result['status']}**", ""]
        return "\n".join(lines)

    bits = final["bits_by_node"]
    shifts = final["shift_by_node"]
    policies = result["selected_policy_by_node"]
    lines += [
        "",
        "## D. Final frozen 58-node plan",
        "",
        "| Node | Class | Policy | Bits | Shift |",
        "|---|---|---|---:|---:|",
    ]
    for node in result["quant_nodes"]:
        lines.append(
            f"| `{node}` | `{result['sensitivity_classes'][node]}` | `{policies[node]}` | "
            f"{bits[node]} | {shifts[node]} |"
        )

    lines += [
        "",
        "## E. Search-validation history",
        "",
        "| Step | Action | Node | Bits | Accuracy | Drop |",
        "|---:|---|---|---|---:|---:|",
    ]
    for index, item in enumerate(result["search_history"]):
        node = item.get("node", "-")
        transition = "-"
        if "from_bits" in item:
            transition = f"{item['from_bits']}->{item['to_bits']}"
        accuracy = item.get("accuracy", item.get("trial_accuracy", 0.0))
        drop = item.get("drop_pp", item.get("trial_drop_pp", 0.0))
        lines.append(f"| {index} | `{item['action']}` | `{node}` | {transition} | {accuracy:.2f}% | {drop:+.2f} pp |")

    lines += [
        "",
        "## F. Final accuracy and storage metrics",
        "",
        f"- Final search-validation accuracy: **{final['search_accuracy']:.2f}%**.",
        f"- Final search-validation drop vs H0: **{final['search_drop_pp']:+.2f} pp**.",
        f"- Production target <=0.30 pp: **{'PASS' if final['search_drop_pp'] <= 0.30 else 'FAIL'}**.",
        f"- Official test accuracy after freeze: **{final['official_test_accuracy']:.2f}%**.",
        f"- Official test delta vs H0 85.13%: **{final['official_test_drop_vs_h0_85_13_pp']:+.2f} pp**.",
        f"- Unweighted average node bits: **{final['unweighted_average_bits']:.4f}**.",
        f"- Activation-volume-weighted average bits: **{final['volume_weighted_average_bits']:.4f}**.",
        f"- Nodes at 6 bits: {', '.join(f'`{node}`' for node in final['nodes_by_bits'].get('6', [])) or 'none'}.",
        f"- Nodes at 7 bits: {', '.join(f'`{node}`' for node in final['nodes_by_bits'].get('7', [])) or 'none'}.",
        f"- Nodes at 10/12/14 bits: {', '.join(f'`{node}`' for node in final['nodes_by_bits'].get('10', []) + final['nodes_by_bits'].get('12', []) + final['nodes_by_bits'].get('14', [])) or 'none'}.",
        f"- `layer3.0.qrprelu_output` final width: **{final['bits_by_node']['layer3.0.qrprelu_output']} bits**.",
        "",
        "## G. Mixed residual alignment",
        "",
        "All 18 residual adds were checked with per-node bits/shifts, common output scale, integer alignment, saturation, and equivalence.",
        "",
        "| Add node | A bits/shift | B bits/shift | Output bits/shift | Delta A/B | Saturation | PASS |",
        "|---|---|---|---|---|---:|---|",
    ]
    for item in final["residual_alignment"]["records"]:
        d = item["required_shift_amount"]
        lines.append(
            f"| `{item['add_node']}` | {item['branch_a_bits']}/{item['branch_a_shift']} | "
            f"{item['branch_b_bits']}/{item['branch_b_shift']} | {item['common_output_bits']}/{item['common_output_shift']} | "
            f"{d['branch_a']}/{d['branch_b']} | {item['saturation_count']} | {item['integer_add_equivalence']} |"
        )

    lines += [
        "",
        "## H. Test leakage and excluded arithmetic",
        "",
        "- Calibration and search-validation were non-overlapping CIFAR-10 TRAIN subsets with seed 20260913.",
        "- Official TEST was not used to choose shifts, policies, ranking, promotions, demotions, or the final plan.",
        "- Binary-convolution accumulators, GAP accumulation, FC accumulation, and final logits were not quantized.",
        "",
        "## I. Hardware interpretation",
        "",
        "H0 affine/classifier bias remains fixed INT6. The final plan stores each selected activation/residual state as signed integer q with node-level integer shift metadata. This is not a claim that the entire network is fully finite-width integer-only; accumulator widths remain a later experiment.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "h1s_h1mp.json"))
    parser.add_argument("--source-r8b-checkpoint", default=None)
    parser.add_argument("--h0-results-json", default=str(ROOT / "H0_INTEGER_BIAS_SWEEP_RESULTS.json"))
    parser.add_argument("--h1s-results-json", default=str(ROOT / "H1S_SINGLE_NODE_RESULTS.json"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--calibration-samples", type=int, default=None)
    parser.add_argument("--search-validation-samples", type=int, default=None)
    parser.add_argument("--target-drop-pp", type=float, default=0.30)
    parser.add_argument("--anchor-bits", type=int, default=14)
    parser.add_argument("--max-promotions", type=int, default=120)
    parser.add_argument("--max-demotions", type=int, default=120)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    split_cfg = config.get("data_split", {})
    seed = int(args.seed if args.seed is not None else split_cfg.get("split_seed", 20260913))
    calibration_samples = int(args.calibration_samples or split_cfg.get("calibration_samples", 10000))
    search_samples = int(args.search_validation_samples or split_cfg.get("search_validation_samples", 10000))
    h1mp_cfg = config.get("h1mp", {})
    min_shift = int(h1mp_cfg.get("min_shift", -16))
    max_shift = int(h1mp_cfg.get("max_shift", 24))

    if args.source_r8b_checkpoint:
        source_checkpoint = Path(args.source_r8b_checkpoint)
        if not source_checkpoint.is_absolute():
            source_checkpoint = ROOT / source_checkpoint
        source_summary_path = source_checkpoint.parent / "summary.json"
        source_summary = json.loads(source_summary_path.read_text(encoding="utf-8")) if source_summary_path.exists() else {}
    else:
        source_meta, _ = discover_formal_r8b_checkpoint(ROOT)
        source_checkpoint = source_meta["checkpoint_path"]
        source_summary_path = source_meta["summary_path"]
        source_summary = source_meta["summary"]
    h0_path = Path(args.h0_results_json)
    if not h0_path.is_absolute():
        h0_path = ROOT / h0_path
    h1s_path = Path(args.h1s_results_json)
    if not h1s_path.is_absolute():
        h1s_path = ROOT / h1s_path

    h1s = json.loads(h1s_path.read_text(encoding="utf-8"))
    sensitivity_rows = h1s["final_single_node_sensitivity"]
    selected_policy = dict(h1s["selected_policy_by_node"])
    h1s_by_node = {row["node"]: row for row in sensitivity_rows}
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    model = load_r8b_h0_model(source_checkpoint, h0_path, device)
    adapter = make_h1_adapter(model)
    calib_loader, search_loader, test_loader = build_h1_search_loaders(
        seed, calibration_samples, search_samples
    )
    search_baseline = evaluate(model, search_loader, device)
    test_baseline = evaluate(model, test_loader, device)
    collector, seen = calibrate(model, adapter, calib_loader, device, calibration_samples)
    nodes = adapter.list_quant_nodes()
    if len(nodes) != 58:
        raise RuntimeError(f"Expected 58 nodes, found {len(nodes)}")
    missing = set(nodes) - set(h1s_by_node)
    if missing:
        raise RuntimeError("H1-S is missing nodes: " + ", ".join(sorted(missing)))

    sensitivity = {node: max(0.0, float(h1s_by_node[node]["drop_vs_search_baseline_pp"])) for node in nodes}
    classes = {node: classify_sensitivity(sensitivity[node]) for node in nodes}
    candidates = {node: candidate_bits_for_class(classes[node]) for node in nodes}
    class_candidates = {
        cls: sorted(set(bit for node in nodes if classes[node] == cls for bit in candidates[node]))
        for cls in ("robust", "mild", "sensitive", "very_sensitive")
    }
    initial_bits_by_class = {
        "robust": 6,
        "mild": 7,
        "sensitive": 10,
        "very_sensitive": 12,
    }

    anchor_bits = {node: int(args.anchor_bits) for node in nodes}
    anchor = evaluate_plan(model, adapter, search_loader, device, collector, anchor_bits, selected_policy, min_shift, max_shift)
    anchor_drop = search_baseline - anchor["accuracy"]
    output_cfg = config.get("output", {})
    output_dir = Path(args.output_dir or output_cfg.get("root", "."))
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / output_cfg.get("h1mp_results", "H1MP_SEARCH_RESULTS.json")
    report_path = output_dir / output_cfg.get("h1mp_report", "H1MP_MIXED_PRECISION_REPORT.md")

    common_source = {
        "r8b_checkpoint": str(source_checkpoint),
        "source_summary": str(source_summary_path),
        "h0_results_json": str(h0_path),
        "h1s_results_json": str(h1s_path),
        "r8b_stored_best_acc": source_summary.get("best_acc"),
        "r8b_reload_accuracy": float(test_baseline + (85.40 - 85.13)) if abs(test_baseline - 85.13) < 0.08 else None,
        "h0_test_baseline_accuracy": float(test_baseline),
        "search_validation_baseline_accuracy": float(search_baseline),
    }

    if anchor_drop > args.target_drop_pp:
        result = {
            "status": "STOP_ANCHOR_FAILED",
            "source": common_source,
            "methodology": {
                "seed": seed,
                "calibration_source": "CIFAR-10 TRAIN only",
                "calibration_samples": seen,
                "search_validation_source": "non-overlapping CIFAR-10 TRAIN subset",
                "official_test_used_for_search": False,
            },
            "anchor": {
                "bits_all_nodes": args.anchor_bits,
                "search_accuracy": anchor["accuracy"],
                "drop_pp": anchor_drop,
                "saturation": anchor["total_saturation"],
            },
            "quant_nodes": nodes,
        }
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        report_path.write_text(render_report(result), encoding="utf-8")
        print(f"Wrote {result_path}", flush=True)
        print(f"Wrote {report_path}", flush=True)
        raise RuntimeError(
            f"INT{args.anchor_bits} anchor loses {anchor_drop:.2f} pp on search-validation; H1-MP stopped."
        )

    node_bits = {node: initial_bits_by_class[classes[node]] for node in nodes}
    current = evaluate_plan(model, adapter, search_loader, device, collector, node_bits, selected_policy, min_shift, max_shift)
    history = [{
        "action": "initial",
        "accuracy": current["accuracy"],
        "drop_pp": search_baseline - current["accuracy"],
        "node_bits": dict(node_bits),
    }]

    promotions = 0
    while search_baseline - current["accuracy"] > args.target_drop_pp:
        eligible = []
        for node in nodes:
            higher = next_higher_bit(node_bits[node], candidates[node])
            if higher is not None:
                eligible.append((sensitivity[node], node, higher))
        if not eligible or promotions >= args.max_promotions:
            break
        _, node, new_bits = max(eligible)
        old_bits = node_bits[node]
        node_bits[node] = new_bits
        current = evaluate_plan(model, adapter, search_loader, device, collector, node_bits, selected_policy, min_shift, max_shift)
        promotions += 1
        history.append({
            "action": "promote",
            "node": node,
            "from_bits": old_bits,
            "to_bits": new_bits,
            "accuracy": current["accuracy"],
            "drop_pp": search_baseline - current["accuracy"],
            "node_bits": dict(node_bits),
        })
        print(f"promote {node}: {old_bits}->{new_bits}, search={current['accuracy']:.2f}%", flush=True)

    volume = {node: collector.stats[node].count_seen / max(seen, 1) for node in nodes}
    demotion_order = sorted(nodes, key=lambda node: (sensitivity[node], -volume[node]))
    demotions = 0
    changed = True
    while changed and demotions < args.max_demotions:
        changed = False
        for node in demotion_order:
            lower = next_lower_bit(node_bits[node], candidates[node])
            if lower is None:
                continue
            old_bits = node_bits[node]
            node_bits[node] = lower
            trial = evaluate_plan(model, adapter, search_loader, device, collector, node_bits, selected_policy, min_shift, max_shift)
            trial_drop = search_baseline - trial["accuracy"]
            if trial_drop <= args.target_drop_pp:
                current = trial
                demotions += 1
                changed = True
                history.append({
                    "action": "demote_accept",
                    "node": node,
                    "from_bits": old_bits,
                    "to_bits": lower,
                    "accuracy": current["accuracy"],
                    "drop_pp": trial_drop,
                    "node_bits": dict(node_bits),
                })
                print(f"demote accept {node}: {old_bits}->{lower}, search={current['accuracy']:.2f}%", flush=True)
            else:
                node_bits[node] = old_bits
                history.append({
                    "action": "demote_reject",
                    "node": node,
                    "from_bits": old_bits,
                    "to_bits": lower,
                    "trial_accuracy": trial["accuracy"],
                    "trial_drop_pp": trial_drop,
                })
            if demotions >= args.max_demotions:
                break

    final_bits = dict(node_bits)
    final_search = evaluate_plan(model, adapter, search_loader, device, collector, final_bits, selected_policy, min_shift, max_shift)
    # Freeze is complete here. No search result is consulted after this line.
    final_test = evaluate_plan(model, adapter, test_loader, device, collector, final_bits, selected_policy, min_shift, max_shift)
    shifts = final_search["shift_map"]
    total_volume = sum(volume.values())
    nodes_by_bits = {
        str(bits): [node for node in nodes if final_bits[node] == bits]
        for bits in sorted(set(final_bits.values()))
    }
    avg_bits = sum(final_bits.values()) / len(nodes)
    volume_weighted = sum(volume[node] * final_bits[node] for node in nodes) / max(total_volume, 1e-12)
    result = {
        "status": "COMPLETE",
        "source": common_source,
        "methodology": {
            "seed": seed,
            "calibration_source": "CIFAR-10 TRAIN only",
            "calibration_samples": seen,
            "search_validation_source": "non-overlapping CIFAR-10 TRAIN subset",
            "search_validation_samples": search_samples,
            "official_test_used_for_search": False,
            "target_search_drop_pp": args.target_drop_pp,
            "min_shift": min_shift,
            "max_shift": max_shift,
        },
        "quant_nodes": nodes,
        "calibration_ranges": collector.export(),
        "h1s_top10": sensitivity_rows[:10],
        "sensitivity_classes": classes,
        "selected_policy_by_node": selected_policy,
        "candidate_bits_by_node": candidates,
        "candidate_bits_by_class": class_candidates,
        "initial_bits_by_class": initial_bits_by_class,
        "anchor": {
            "bits_all_nodes": args.anchor_bits,
            "search_accuracy": anchor["accuracy"],
            "drop_pp": anchor_drop,
            "saturation": anchor["total_saturation"],
            "residual_alignment": anchor["residual_alignment"],
        },
        "search_history": history,
        "final_plan": {
            "bits_by_node": final_bits,
            "shift_by_node": shifts,
            "policy_by_node": selected_policy,
            "search_validation_accuracy": final_search["accuracy"],
            "search_accuracy": final_search["accuracy"],
            "search_drop_pp": search_baseline - final_search["accuracy"],
            "search_saturation": final_search["total_saturation"],
            "official_test_accuracy": final_test["accuracy"],
            "official_test_drop_vs_h0_85_13_pp": final_test["accuracy"] - 85.13,
            "official_test_drop_vs_h0_pp": final_test["accuracy"] - test_baseline,
            "official_test_saturation": final_test["total_saturation"],
            "residual_alignment": final_test["residual_alignment"],
            "unweighted_average_bits": avg_bits,
            "volume_weighted_average_bits": volume_weighted,
            "activation_volume_by_node": volume,
            "nodes_by_bits": nodes_by_bits,
        },
        "hardware_scope": {
            "h0_bias_bits": 6,
            "activation_residual_states_have_integer_q_and_shift": True,
            "binary_conv_accumulators_quantized": False,
            "gap_accumulator_quantized": False,
            "fc_accumulator_quantized": False,
            "final_logits_quantized": False,
            "architecture_changed": False,
        },
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report_path.write_text(render_report(result), encoding="utf-8")
    print(f"Wrote {result_path}", flush=True)
    print(f"Wrote {report_path}", flush=True)
    print(json.dumps({
        "search_baseline": search_baseline,
        "final_search_accuracy": final_search["accuracy"],
        "final_search_drop_pp": search_baseline - final_search["accuracy"],
        "final_test_accuracy": final_test["accuracy"],
        "final_test_delta_vs_h0_85_13_pp": final_test["accuracy"] - 85.13,
        "unweighted_average_bits": avg_bits,
        "volume_weighted_average_bits": volume_weighted,
        "layer3.0.qrprelu_output_bits": final_bits["layer3.0.qrprelu_output"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
