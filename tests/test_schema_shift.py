"""Reshaping a Choice's options must never move the right answer."""

from __future__ import annotations

import random

import pytest

from trigon.evals.harness import Case, Expectation
from trigon.evals.schema_shift import (
    COMPAT_MAX_OPTIONS,
    RENAMES,
    Reshape,
    reshape_all,
    reshape_case,
)
from trigon.types import DecisionRequest

NAMES = [f"intent_{i:02d}_thing" for i in range(77)]


def _case(label: int | None = 41, distribution: tuple[float, ...] | None = None) -> Case:
    request = DecisionRequest.model_validate(
        {
            "state": "my card never arrived",
            "questions": {
                "intent": {
                    "type": "choice",
                    "instructions": "Which intent?",
                    "options": [{"name": n, "criteria": f"about {n}"} for n in NAMES],
                },
                "severity": {
                    "type": "score",
                    "instructions": "How bad?",
                    "levels": [{"name": "low"}, {"name": "mid"}, {"name": "high"}],
                },
                "urgent": {"type": "noul", "instructions": "Urgent?"},
            },
        }
    )
    return Case(
        case_id="c",
        request=request,
        expected={
            "intent": Expectation(label=label, distribution=distribution),
            "severity": Expectation(label=2),
            "urgent": Expectation(probability=1.0),
        },
    )


def _truth(case: Case) -> str:
    options = case.request.questions["intent"].options
    return options[case.expected["intent"].hard_label].criteria


@pytest.mark.parametrize("seed", range(25))
def test_the_label_follows_its_option(seed):
    spec = Reshape(min_options=2, max_options=COMPAT_MAX_OPTIONS, rename=0.5, criteria_only=0.2)
    before = _case()
    after = reshape_case(before, random.Random(seed), spec)
    options = after.request.questions["intent"].options
    assert 2 <= len(options) <= COMPAT_MAX_OPTIONS
    # Criteria are never rewritten except to prepend the name, so they identify
    # the option across every surface form the name can take.
    assert _truth(after).endswith(_truth(before))
    assert len({o.name for o in options}) == len(options)


def test_a_distribution_is_restricted_and_renormalised():
    mass = [0.0] * 77
    mass[41], mass[3], mass[60] = 0.6, 0.3, 0.1
    before = _case(label=None, distribution=tuple(mass))
    after = reshape_case(before, random.Random(0), Reshape(max_options=10))
    kept = [o.name for o in after.request.questions["intent"].options]
    dist = after.expected["intent"].distribution
    assert pytest.approx(sum(dist)) == 1.0
    assert kept[max(range(len(dist)), key=dist.__getitem__)] == NAMES[41]
    assert after.expected["intent"].label is None


def test_scores_and_nouls_pass_through():
    before = _case()
    after = reshape_case(before, random.Random(1), Reshape(max_options=5, rename=1.0))
    assert after.request.questions["severity"] == before.request.questions["severity"]
    assert after.request.questions["urgent"] == before.request.questions["urgent"]
    assert after.expected["severity"] == before.expected["severity"]


def test_an_unlabelled_choice_keeps_every_option():
    before = _case()
    before = Case(case_id="c", request=before.request, expected={})
    after = reshape_case(before, random.Random(2), Reshape(max_options=5))
    assert len(after.request.questions["intent"].options) == 77


def test_the_evaluation_form_is_reproducible():
    cases = [_case(label=i) for i in range(10)]
    spec = Reshape(max_options=20, rename=0.5)
    first, second = reshape_all(cases, spec, seed=7), reshape_all(cases, spec, seed=7)
    assert [c.request for c in first] == [c.request for c in second]


def test_renames_keep_the_words():
    for name, rename in RENAMES.items():
        assert rename("card_not_Working").lower().replace("-", " ").replace("_", " ") == (
            "card not working"
        ), name


def test_the_crossover_fraction_trains_both_scoring_heads():
    """Half the reshapes above the crossover, half at or below it."""
    from trigon.schema.compiler import DOT_PRODUCT_CROSSOVER

    spec = Reshape(min_options=2, max_options=77, crossover_fraction=0.5)
    rng = random.Random(0)
    sizes = [
        len(reshape_case(_case(), rng, spec).request.questions["intent"].options)
        for _ in range(400)
    ]
    above = sum(size > DOT_PRODUCT_CROSSOVER for size in sizes) / len(sizes)
    assert 0.4 < above < 0.6
    assert min(sizes) >= 2 and max(sizes) <= 77
    # Without it, a uniform draw puts only 13 of 76 sizes above.
    plain = [
        len(reshape_case(_case(), rng, Reshape(max_options=77)).request.questions["intent"].options)
        for _ in range(400)
    ]
    assert sum(size > DOT_PRODUCT_CROSSOVER for size in plain) / len(plain) < 0.3
