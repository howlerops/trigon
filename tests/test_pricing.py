"""The cost model's arithmetic, and the fairness of its baseline.

`scripts/price.py` is how a use case gets quoted, and a pricing tool that
flatters its own side is worse than no pricing tool — it produces a number
people repeat. The first draft did exactly that: it compared a *cached* typed
path against an *uncached* prompted one, which inflated the break-even from
~2.1× to ~3.4×. These pin the properties that stop it happening again.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

from trigon.schema import SchemaCompiler
from trigon.types import SystemOneRequest
from trigon.usecases import all_use_cases, use_case

PRICE = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "price.py"
_spec = importlib.util.spec_from_file_location("price", PRICE)
price = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(price)


def test_every_use_case_compiles_within_budget():
    """A use case that does not compile is a document, not a schema."""
    compiler = SchemaCompiler()
    for item in all_use_cases():
        compiled = compiler.compile_request(
            SystemOneRequest(state=item.state, questions=item.questions)
        )
        assert compiled.total_tokens > 0
        assert compiled.total_tokens <= compiled.budget.context_tokens
        # Each carries all three primitives, which is the point of having them:
        # a contract demonstrated on one primitive is not demonstrated.
        kinds = {q.kind for q in compiled.schema.questions}
        assert kinds == {"choice", "score", "noul"}, f"{item.name} exercises only {kinds}"


def test_each_use_case_names_a_green_corpus():
    """A use case without a corpus is a demo.

    `docs/data.md` tiers every corpus, and only green may be trained on. Naming
    one is what makes a use case a claim about something buildable.
    """
    audit = (pathlib.Path(__file__).resolve().parent.parent / "docs" / "data.md").read_text()
    for item in all_use_cases():
        assert item.corpus, f"{item.name} names no corpus"
        # The corpus line names at least one dataset the audit actually cleared.
        cleared = [
            line.split("|")[1].strip()
            for line in audit.splitlines()
            if line.startswith("|") and "green" in line
        ]
        assert any(name.split()[0] in item.corpus for name in cleared if name), (
            f"{item.name} cites no corpus the audit cleared to green: {item.corpus}"
        )


def test_the_prompted_baseline_splits_what_billing_splits():
    """The prefix is reusable and the per-request part is not.

    If the split is wrong the whole comparison is wrong, and wrong in our
    favour: pricing the prompted path's instruction block on every request
    while caching ours is scoring the alternative with a feature switched off.
    """
    item = use_case("sentiment")
    prefix, per_request, reply = price._prompt_for(item)

    # The per-request part is the state and nothing else.
    assert "Ordered the 512GB model" in per_request
    assert "Reply with JSON only" not in per_request

    # The prefix carries the instructions and every label, and no state.
    assert "Reply with JSON only" in prefix
    for label in ("positive", "negative", "mixed", "neutral"):
        assert label in prefix
    assert "Ordered the 512GB model" not in prefix

    # The reply is the answer shape, one field per question.
    import json as _json

    assert set(_json.loads(reply)) == set(item.questions)


def test_the_baseline_answers_every_question_in_one_call():
    """Not N calls. A baseline chosen to lose is not a baseline.

    The prompted path's whole cost disadvantage would be manufactured by
    charging it once per question for re-reading the same state.
    """
    item = use_case("moderation")
    prefix, _, reply = price._prompt_for(item)
    import json as _json

    assert len(_json.loads(reply)) == len(item.questions) == 3
    assert prefix.count("Reply with JSON only") == 1


@pytest.mark.parametrize("name", [u.name for u in all_use_cases()])
def test_the_typed_advantage_is_modest_on_tokens_alone(name):
    """The finding `docs/pricing.md` leads with, pinned so it cannot drift.

    Compared like for like, the typed path sends about as many tokens as the
    prompted one — on `moderation` it sends more. The advantage that survives
    is that it generates nothing. So token layout justifies roughly a 2× price
    premium, and anything beyond that has to come from $/token, which this
    project has not measured.

    Asserted as a band rather than a number: the point is the order of
    magnitude, and a test that breaks on a one-token change in a docstring
    would just get deleted.
    """
    from trigon.bpe import BPETokenizer
    from trigon.schema.tokens import CallableEstimator

    item = use_case(name)
    tokenizer_for_compiler = BPETokenizer.load()
    # The same tokenizer both sides, which is the other way this comparison
    # gets rigged: the compiler's default is a character heuristic, and
    # counting the baseline exactly against our estimate inflated the
    # prompted column by about 40%.
    compiled = SchemaCompiler(
        estimator=CallableEstimator(tokenizer_for_compiler.encode, exact=True)
    ).compile_request(SystemOneRequest(state=item.state, questions=item.questions))
    tokenizer = BPETokenizer.load()
    prefix, per_request, reply = price._prompt_for(item)

    typed_cached = compiled.state_tokens + compiled.readout_tokens
    prompted_cached = tokenizer.count(per_request)

    # The exact relationship the doc leads with: both paths read the same
    # state, and the typed path then pays one readout slot per question. If
    # this stops holding, the headline is describing something else.
    assert typed_cached == prompted_cached + compiled.readout_tokens
    break_even = (prompted_cached + tokenizer.count(reply) * 4.0) / typed_cached

    assert 1.5 < break_even < 5.0, (
        f"{name} break-even {break_even:.2f}x -- if this has moved far, "
        "docs/pricing.md leads with a number that is no longer true"
    )
    # Caching both sides is what keeps it honest; caching only ours inflates it.
    unfair = (
        tokenizer.count(prefix) + prompted_cached + tokenizer.count(reply) * 4.0
    ) / typed_cached
    assert unfair > break_even
