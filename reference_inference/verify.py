"""Dual inference verifier. PyTorch appears only on this oracle side."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from recu_hw.h2_workspace import build_h2_trace_adapter, load_r8b_h1mp_model
from .reference_inference import IntegerReference


ROOT = Path(__file__).resolve().parents[1]


def load_official_test():
    with (ROOT / "data/cifar-10-batches-py/test_batch").open("rb") as handle:
        raw = pickle.load(handle, encoding="bytes")
    return raw[b"data"].reshape(-1, 3, 32, 32).astype(np.uint8), np.asarray(raw[b"labels"], dtype=np.int64)


def make_oracle():
    config = json.loads((ROOT / "reference_inference/model_config.json").read_text(encoding="utf-8"))
    model = load_r8b_h1mp_model(ROOT / config["checkpoint"], ROOT / "H0_INTEGER_BIAS_SWEEP_RESULTS.json", ROOT / "H1MP_SEARCH_RESULTS.json", torch.device("cpu"))
    adapter = build_h2_trace_adapter(model)
    h2 = json.loads((ROOT / "H2_FINITE_WIDTH_RESULTS.json").read_text(encoding="utf-8"))
    adapter.set_h2_plan(h2["h2a_plan"])
    adapter.set_trace_enabled(True)
    return adapter


def oracle_one(adapter, rgb, capture=True):
    x = torch.from_numpy(rgb.astype(np.float32) / np.float32(255.0))
    observed = {}
    original = adapter._record

    def record(name, value, *args, **kwargs):
        if capture and name in adapter.capture_names:
            observed[name] = value.detach().cpu().numpy().copy()
        return original(name, value, *args, **kwargs)

    adapter._record = record
    try:
        with torch.no_grad():
            if capture:
                observed["thermometer"] = adapter.model.thermo(x).detach().cpu().numpy().astype(np.int8)
            logits = adapter(x)
    finally:
        adapter._record = original
    if capture:
        for prefix in (f"layer{s}.{i}" for s in (1, 2, 3) for i in range(3)):
            name = f"h1.{prefix}.second_add"
            observed[f"qrprelu.{prefix}.branch_positive"] = (observed[name] >= 0).astype(np.uint8)
            observed[f"{prefix}.stage_output"] = observed[f"h1.{prefix}.qrprelu_output"]
    return logits.argmax(dim=1).cpu().numpy(), observed


def compare_nodes(reference, oracle):
    differences = {}
    for name, expected in reference.items():
        if name not in oracle:
            differences[name] = {"status": "missing_oracle_node"}
            continue
        actual = oracle[name]
        if actual.shape != expected.shape:
            differences[name] = {"status": "shape_mismatch", "expected": list(expected.shape), "actual": list(actual.shape)}
            continue
        diff = np.abs(actual.astype(np.int64) - expected.astype(np.int64))
        differences[name] = {"max_abs_error": int(diff.max()) if diff.size else 0,
                             "mismatch_count": int(np.count_nonzero(diff))}
    return differences


def verify(counts=(1, 10, 100), full=False):
    images, labels = load_official_test()
    ref = IntegerReference()
    oracle = make_oracle()
    # Capture exactly the nodes exposed by the standalone reference.
    oracle.capture_names = set()
    start = ref.forward_uint8(images[:1], trace=True)
    oracle.capture_names = set(ref.trace) - {"thermometer", *[f"layer{s}.{i}.stage_output" for s in (1, 2, 3) for i in range(3)]}
    result = {"small_batches": {}, "node_errors": {}, "thermometer_mismatch": 0,
              "qrp_branch_mismatch": 0, "maximum_integer_node_error": 0,
              "final_logits_mismatch": 0, "prediction_mismatch": 0,
              "overflow_count": 0, "h1_saturation_count": 0,
              "official_test_accuracy_percent": None,
              "current_implementation_test_accuracy_percent": None}
    for count in counts:
        differences = {}
        pred_errors = 0
        for index in range(count):
            ref_logits = ref.forward_uint8(images[index], trace=True)
            result["overflow_count"] += sum(value for name, value in ref.overflow.items() if not name.startswith("h1."))
            result["h1_saturation_count"] += sum(value for name, value in ref.overflow.items() if name.startswith("h1."))
            oracle_pred, observed = oracle_one(oracle, images[index:index + 1], capture=True)
            errors = compare_nodes(ref.trace, observed)
            for name, item in errors.items():
                previous = differences.get(name)
                if previous is None:
                    differences[name] = dict(item)
                elif "max_abs_error" in item:
                    previous["max_abs_error"] = max(previous["max_abs_error"], item["max_abs_error"])
                    previous["mismatch_count"] += item["mismatch_count"]
            pred_errors += int(ref_logits.argmax(axis=1)[0] != oracle_pred[0])
        worst = max((item.get("max_abs_error", 0) for item in differences.values()), default=0)
        result["small_batches"][str(count)] = {"node_count": len(differences),
                                               "maximum_integer_node_error": worst,
                                               "prediction_mismatch": pred_errors}
        result["node_errors"] = differences
        result["maximum_integer_node_error"] = max(result["maximum_integer_node_error"], worst)
        result["thermometer_mismatch"] += differences.get("thermometer", {}).get("mismatch_count", 0)
        result["qrp_branch_mismatch"] += sum(item.get("mismatch_count", 0) for name, item in differences.items() if name.endswith(".branch_positive"))
        result["final_logits_mismatch"] += differences.get("final_logits", {}).get("mismatch_count", 0)
        result["prediction_mismatch"] += pred_errors
        print(f"verified {count}: nodes={len(differences)} max_error={worst} pred_mismatch={pred_errors}", flush=True)
        if worst or pred_errors or any(item.get("status") for item in differences.values()) or result["overflow_count"]:
            break
    else:
        if full:
            oracle.set_trace_enabled(False)
            oracle.capture_names = set()
            correct = 0
            oracle_correct = 0
            for start in range(0, len(images), 16):
                end = min(start + 16, len(images))
                ref_logits = ref.forward_uint8(images[start:end], trace=False)
                result["overflow_count"] += sum(value for name, value in ref.overflow.items() if not name.startswith("h1."))
                result["h1_saturation_count"] += sum(value for name, value in ref.overflow.items() if name.startswith("h1."))
                predictions, _ = oracle_one(oracle, images[start:end], capture=False)
                result["prediction_mismatch"] += int(np.count_nonzero(ref_logits.argmax(axis=1) != predictions))
                correct += int(np.count_nonzero(ref_logits.argmax(axis=1) == labels[start:end]))
                oracle_correct += int(np.count_nonzero(predictions == labels[start:end]))
                if end % 1000 == 0:
                    print(f"official test {end}/10000", flush=True)
            result["official_test_accuracy_percent"] = 100.0 * correct / len(images)
            result["current_implementation_test_accuracy_percent"] = 100.0 * oracle_correct / len(images)
    output = ROOT / "reference_inference/verification_results.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    r = verify(full=args.full)
    print(json.dumps({key: value for key, value in r.items() if key != "node_errors"}, indent=2))
