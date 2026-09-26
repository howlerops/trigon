"""Faithfulness: comprehensiveness and sufficiency, and the controls beside them.

`trigon.evals.faithfulness` is the measurement `docs/decisions.md` names in
*Plausibility is the wrong target*. A faithfulness metric the controls pass
measures nothing, so the first tests here construct a backend whose answer
depends on exactly one word and check that the metric can tell the highlighter
that names it from one that guesses. The rest pin the arithmetic: which words
a bin takes, what deleting them leaves, and the AOPC over the bins.
"""

from __future__ import annotations

import math

import pytest

from trigon.backends.base import BackendOutput, QuestionOutput
from trigon.engine import Engine
from trigon.evals.faithfulness import (
    FAITHFULNESS_BINS,
    Prober,
    aopc,
    bin_count,
    delete_words,
    engine_scorer,
    faithfulness,
    random_scorer,
    render_faithfulness,
    run_faithfulness_suite,
    top_words,
    word_scores,
)
from trigon.evals.harness import Case, Expectation
from trigon.evals.rationale import rationale_cases, words
from trigon.schema import render_state
from trigon.types import DecisionRequest, NoulQuestion


class _Marked:
    """A Noul backend whose probability is a known function of which words are present.

    ``p(yes) = base + sum(weight of each marker word in the state)``, returned
    as the logit that produces it. Evidence, when asked for, scores each word
    by ``evidence[word]`` (0 for any other), as per-token triples -- one token
    per word, with a leading space the way a byte-level tokenizer returns one.
    Every state it is asked about is recorded, so a test can see that removal
    really re-asked.
    """

    model_version = "marked-0"

    def __init__(self, base: float, weights: dict[str, float], evidence: dict[str, float]):
        self.base, self.weights, self.evidence = base, weights, evidence
        self.seen: list[str] = []

    def infer(self, compiled, request):
        text = render_state(request.state)
        self.seen.append(text)
        present = {text[s:e] for s, e in words(text)}
        p = self.base + sum(w for word, w in self.weights.items() if word in present)
        logit = math.log(p / (1 - p))
        evidence = None
        if request.options.include_evidence:
            evidence = tuple(
                (max(0, s - 1), e, self.evidence.get(text[s:e], 0.0)) for s, e in words(text)
            )
        return BackendOutput(
            outputs={
                q.question_id: QuestionOutput(
                    question_id=q.question_id,
                    kind=q.kind,
                    logits=(logit,),
                    evidence=evidence,
                    evidence_method="gradient_x_input" if evidence is not None else None,
                )
                for q in compiled.schema.questions
            },
            model_version=self.model_version,
        )


def _case(i: int, state: str, marked: str = "zorp") -> Case:
    at = state.index(marked)
    return Case(
        case_id=f"f/{i}",
        request=DecisionRequest(
            state=state, questions={"toxic": NoulQuestion(instructions="Is this toxic?")}
        ),
        expected={"toxic": Expectation(probability=1.0, rationale=((at, at + len(marked)),))},
    )


def _posts(n: int = 40) -> list[Case]:
    """Twenty-word posts with the one word that matters at a different place in each."""
    cases = []
    for i in range(n):
        filler = [f"w{j}" for j in range(19)]
        filler.insert(i % 20, "zorp")
        cases.append(_case(i, " ".join(filler)))
    return cases


# -- the metric separates a highlighter from a guess --------------------------


@pytest.fixture
def one_word():
    """p(yes) is 0.98 with "zorp" in the state and 0.02 without, and nothing else counts."""
    backend = _Marked(base=0.02, weights={"zorp": 0.96}, evidence={"zorp": 1.0})
    return backend, Engine(backend)


def test_the_word_the_answer_depends_on_is_comprehensive_and_sufficient(one_word):
    backend, engine = one_word
    pairs = rationale_cases(_posts())
    prober = Prober(engine)
    named = faithfulness("oracle", engine_scorer(engine), pairs, prober)
    guessed = faithfulness("random", random_scorer(0), pairs, prober)

    # Remove "zorp" and the answer collapses; keep only "zorp" and it stands.
    assert named.comprehensiveness == pytest.approx(0.96, abs=1e-9)
    assert named.sufficiency == pytest.approx(0.0, abs=1e-9)
    assert named.method == "gradient_x_input"
    # A random word is "zorp" one time in twenty at 1%, and half the time at 50%.
    assert guessed.comprehensiveness < 0.4
    assert guessed.sufficiency > 0.6
    assert guessed.comprehensiveness_by_bin[0] < guessed.comprehensiveness_by_bin[-1]


def test_the_suite_scores_the_controls_beside_the_model_and_pairs_them(one_word):
    _, engine = one_word
    training = _posts(10)
    rows, prober = run_faithfulness_suite(
        {"model": engine_scorer(engine)}, engine, _posts(), training, n=30
    )
    assert [r.name for r in rows] == ["model", "rationale lexicon", "random"]
    by = {r.name: r for r in rows}
    assert all(r.n == 30 for r in rows)
    # The lexicon learned "zorp" from the training highlights, so on this
    # backend it is as faithful as the oracle: the control a reading must beat.
    assert by["rationale lexicon"].comprehensiveness == pytest.approx(0.96, abs=1e-9)
    assert by["model"].extra["comprehensiveness_minus_random"] > 0.5
    assert by["model"].extra["sufficiency_minus_random"] < -0.5
    assert "comprehensiveness_minus_random" not in by["random"].extra
    # One forward per distinct text: the three methods share their probes.
    assert prober.forwards < 30 * 3 * (1 + 2 * len(FAITHFULNESS_BINS))
    table = render_faithfulness(rows)
    assert "| random |" in table and "50%" in table


def test_removal_is_a_real_re_ask_of_the_shorter_state(one_word):
    backend, engine = one_word
    case = _case(0, "a zorp b c")
    faithfulness("oracle", engine_scorer(engine), rationale_cases([case]), Prober(engine))
    assert "a b c" in backend.seen  # zorp removed
    assert "zorp" in backend.seen  # zorp alone


# -- the arithmetic -----------------------------------------------------------


def test_deleting_words_keeps_the_whitespace_that_followed_each_kept_word():
    text = "  one two\nthree  four "
    units = words(text)
    assert delete_words(text, units, set()) == text
    assert delete_words(text, units, {1}) == "  one three  four "
    assert delete_words(text, units, {0}) == "  two\nthree  four "
    assert delete_words(text, units, {3}) == "  one two\nthree "
    assert delete_words(text, units, {0, 1, 2, 3}) == "   "


def test_a_bin_takes_the_ceiling_of_its_share_and_at_least_one_word():
    assert bin_count(0.01, 20) == 1
    assert bin_count(0.10, 30) == 3  # 0.1 * 30 is 3.0000000000000004
    assert bin_count(0.20, 12) == 3
    assert bin_count(0.50, 1) == 1
    assert bin_count(0.50, 0) == 0


def test_ties_are_broken_by_the_draw_and_not_by_position():
    scores = [0.0, 1.0, 0.0, 0.0]
    assert top_words(scores, 1, [0.9, 0.9, 0.9, 0.9]) == {1}
    assert top_words(scores, 2, [0.9, 0.5, 0.8, 0.1]) == {1, 3}


def test_a_word_scores_its_best_token_and_a_leading_space_scores_nothing():
    text = "the unlikely word"
    tokens = [(0, 3, 0.2), (3, 6, 0.1), (6, 12, 0.7), (12, 17, 0.4)]
    # " un" touches "unlikely" only; "likely" too; " word" touches "word".
    assert word_scores(text, tokens) == [0.2, 0.7, 0.4]
    assert word_scores(text, [(3, 4, 0.9)]) == [0.0, 0.0, 0.0]


def test_aopc_is_the_mean_over_bins_on_a_worked_example():
    """p(yes) = 0.5 + 0.1 [alpha] + 0.3 [zorp]; zorp ranked first, alpha second.

    Ten words, so the bins take 1, 1, 1, 2 and 5 words. Removing zorp alone
    costs 0.3, removing both costs 0.4: comprehensiveness (3 x 0.3 + 2 x 0.4) / 5
    = 0.34. Keeping zorp alone costs 0.1, keeping both costs nothing:
    sufficiency 3 x 0.1 / 5 = 0.06.
    """
    backend = _Marked(
        base=0.5, weights={"zorp": 0.3, "alpha": 0.1}, evidence={"zorp": 1.0, "alpha": 0.5}
    )
    engine = Engine(backend)
    case = _case(0, "zorp alpha b c d e f g h i")
    row = faithfulness("worked", engine_scorer(engine), rationale_cases([case]), Prober(engine))
    assert [bin_count(k, 10) for k in FAITHFULNESS_BINS] == [1, 1, 1, 2, 5]
    assert row.full_probability == pytest.approx(0.9, abs=1e-9)
    assert row.comprehensiveness_by_bin == pytest.approx((0.3, 0.3, 0.3, 0.4, 0.4), abs=1e-9)
    assert row.sufficiency_by_bin == pytest.approx((0.1, 0.1, 0.1, 0.0, 0.0), abs=1e-9)
    assert row.comprehensiveness == pytest.approx(0.34, abs=1e-9)
    assert row.sufficiency == pytest.approx(0.06, abs=1e-9)
    assert aopc([0.3, 0.3, 0.3, 0.4, 0.4]) == pytest.approx(0.34)


def test_a_backend_that_cannot_attribute_is_an_error_not_a_row_of_zeros():
    class _Silent(_Marked):
        def infer(self, compiled, request):
            out = super().infer(compiled, request)
            return BackendOutput(
                outputs={
                    k: QuestionOutput(question_id=k, kind=v.kind, logits=v.logits)
                    for k, v in out.outputs.items()
                },
                model_version=out.model_version,
            )

    engine = Engine(_Silent(base=0.5, weights={}, evidence={}))
    with pytest.raises(ValueError, match="no evidence"):
        faithfulness("x", engine_scorer(engine), rationale_cases(_posts(1)), Prober(engine))
