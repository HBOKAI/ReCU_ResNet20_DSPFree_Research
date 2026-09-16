"""Focused checks for the independent frozen H2A-v2 integer reference."""

import hashlib
import json
from pathlib import Path

import numpy as np

from reference_inference.binary_conv import binary_conv
from reference_inference.integer_ops import align_away, align_even
from reference_inference.reference_inference import IntegerReference, thermometer_uint8
from reference_inference.verify import load_official_test, make_oracle, oracle_one, compare_nodes


ROOT = Path(__file__).resolve().parents[1]


def test_thermometer_all_256_levels_matches_formal_encoder():
    import torch
    from recu_hw.r5t import ThermometerEncoder

    levels = np.tile(np.arange(256, dtype=np.uint8), 4).reshape(32, 32)
    image = np.stack((levels, np.flipud(levels), np.fliplr(levels)))
    actual = thermometer_uint8(image)[0]
    with torch.no_grad():
        expected = ThermometerEncoder(8)(torch.from_numpy(image.astype(np.float32) / np.float32(255.0))[None]).numpy()[0]
    assert actual.shape == (96, 32, 32)
    assert np.count_nonzero(actual != expected) == 0


def test_binary_xnor_popcount_with_padding_and_zero_activation():
    x = np.array([[[[1, -1, 0], [-1, 1, -1], [0, 1, -1]],
                   [[-1, 1, 1], [0, -1, 1], [1, 0, -1]]]], dtype=np.int8)
    w = np.array([[[[1, -1, 1], [-1, 1, -1], [1, 1, -1]],
                   [[-1, 1, -1], [1, -1, 1], [-1, 1, 1]]]], dtype=np.int8)
    actual = binary_conv(x, w)
    expected = np.zeros_like(actual)
    for y in range(3):
        for col in range(3):
            matches = 0
            active = 0
            for channel in range(2):
                for ky in range(3):
                    for kx in range(3):
                        iy, ix = y + ky - 1, col + kx - 1
                        if 0 <= iy < 3 and 0 <= ix < 3 and x[0, channel, iy, ix] != 0:
                            active += 1
                            matches += int(x[0, channel, iy, ix] == w[0, channel, ky, kx])
            expected[0, 0, y, col] = 2 * matches - active
    np.testing.assert_array_equal(actual, expected)


def test_rounding_policies_are_distinct_and_exact():
    values = np.array([-5, -3, -1, 1, 3, 5], dtype=np.int64)
    np.testing.assert_array_equal(align_away(values, 2, 1), [-3, -2, -1, 1, 2, 3])
    np.testing.assert_array_equal(align_even(values, 2, 1), [-2, -2, 0, 0, 2, 2])


def test_export_manifest_and_freeze_hashes():
    config = json.loads((ROOT / "reference_inference/model_config.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "reference_inference/params_export/manifest.json").read_text(encoding="utf-8"))
    with np.load(ROOT / "reference_inference/params_export/inference_params.npz") as arrays:
        assert len(manifest) == len(arrays.files) == 136
        for entry in manifest:
            name = entry["name"]
            assert list(arrays[name].shape) == entry["shape"]
            assert str(arrays[name].dtype) == entry["dtype"]
        binary = [name for name in arrays.files if name.endswith(".weight_sign") and name != "linear.weight_sign"]
        assert sum(arrays[name].size for name in binary) == 281088
        assert all(np.count_nonzero(arrays[name] == 0) == 0 for name in binary)
    for relative, expected in config["source_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected


def test_all_integer_nodes_match_formal_h2a_on_one_image():
    images, _ = load_official_test()
    reference = IntegerReference()
    logits = reference.forward_uint8(images[0], trace=True)
    oracle = make_oracle()
    oracle.capture_names = set(reference.trace)
    oracle_pred, observed = oracle_one(oracle, images[:1])
    differences = compare_nodes(reference.trace, observed)
    assert len(differences) == 245
    assert all(item.get("max_abs_error") == 0 for item in differences.values())
    assert int(logits.argmax(axis=1)[0]) == int(oracle_pred[0])
    assert sum(value for name, value in reference.overflow.items() if not name.startswith("h1.")) == 0


def test_predictions_match_formal_h2a_on_ten_images():
    images, _ = load_official_test()
    reference = IntegerReference()
    oracle = make_oracle()
    oracle.capture_names = set()
    oracle.set_trace_enabled(False)
    expected, _ = oracle_one(oracle, images[:10], capture=False)
    actual = reference.predict_uint8(images[:10])
    np.testing.assert_array_equal(actual, expected)
