"""The parity check fails a deployment that is not the evaluated model."""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))

from serving_parity import compare  # noqa: E402


def _r(p_yes: float, choice: dict[str, float]) -> dict:
    return {
        "answers": {
            "n": {"type": "noul", "probability": p_yes},
            "c": {"type": "choice", "probabilities": choice},
        }
    }


def test_rounding_passes_and_a_flipped_answer_counts():
    a = [_r(0.80, {"x": 0.7, "y": 0.3})]
    near = [_r(0.81, {"x": 0.69, "y": 0.31})]
    flipped = [_r(0.40, {"x": 0.7, "y": 0.3})]
    same = compare(a, near)
    assert same["argmax_agreement"] == 1.0 and same["max_abs_diff"] == pytest.approx(0.01)
    assert compare(a, flipped)["argmax_agreement"] == 0.5


def test_different_labels_are_an_error():
    with pytest.raises(ValueError):
        compare([_r(0.5, {"x": 1.0})], [_r(0.5, {"z": 1.0})])
