"""Evidence: the spans of the state that drove an answer.

Independence of evidence across questions is the architectural claim and
lives in `tests/test_independence.py` with the others. This file holds the
rest: the contract is opt-in and the default response does not change; the
merge rule; token offsets that are exact against the text; plausibility
metrics that mean what the literature means by them; and rationale
supervision that actually switches a checkpoint's evidence to the span head.
"""

from __future__ import annotations

import dataclasses

import pytest
from fastapi.testclient import TestClient

from trigon.backends.base import BackendOutput, QuestionOutput, validate_output
from trigon.backends.lexical import LexicalBackend
from trigon.backends.tokenizer import HashingTokenizer
from trigon.bpe import BPETokenizer
from trigon.engine import Engine
from trigon.evals.harness import Case, Expectation
from trigon.evals.rationale import (
    RationaleLexicon,
    highlighted_words,
    iou_f1,
    rationale_cases,
    run_rationale_suite,
    token_prf,
    words,
)
from trigon.evidence import merge_spans, token_labels
from trigon.schema import compile_request
from trigon.server.app import build_app
from trigon.server.config import ServerConfig
from trigon.types import EvidenceSpan, NoulQuestion, SystemOneRequest

SAMPLES = [
    "",
    "  leading and trailing  ",
    "the card payment was declined at the till",
    '{"plan": "enterprise", "open_tickets": 7}',
    "café ☕ 日本語 ümlauts",
    "emoji 🎯 then more",
    "snake_case camelCase 12,345.67 don't",
    "é combining, and a tab\there",
]


# -- the contract -------------------------------------------------------------


def test_the_default_response_is_unchanged(engine, support_request):
    """Opt-in: a client that never heard of evidence sees no new keys."""
    for answer in engine.answer(support_request).answers.values():
        dumped = answer.model_dump(exclude_none=True)
        assert "evidence" not in dumped and "evidence_method" not in dumped


def test_asking_for_evidence_returns_spans_of_the_callers_own_string(engine, support_request):
    request = support_request.model_copy(
        update={"options": support_request.options.model_copy(update={"include_evidence": True})}
    )
    response = engine.answer(request)
    for answer in response.answers.values():
        assert answer.evidence_method == "lexical_overlap"
        for span in answer.evidence:
            assert support_request.state[span.start : span.end] == span.text
    assert any(a.evidence for a in response.answers.values())


def test_an_empty_span_is_not_a_span():
    with pytest.raises(ValueError):
        EvidenceSpan(start=4, end=4, text="", score=0.5)
    with pytest.raises(ValueError):
        EvidenceSpan(start=0, end=2, text="ab", score=1.5)


class _Silent:
    """A backend that answers and cannot attribute."""

    model_version = "silent-0"

    def infer(self, compiled, request):
        return BackendOutput(
            outputs={
                q.question_id: QuestionOutput(question_id=q.question_id, kind=q.kind, logits=(0.0,))
                for q in compiled.schema.questions
            },
            model_version=self.model_version,
        )


def test_a_backend_that_cannot_attribute_says_so():
    """An empty list alone would read as "nothing mattered"."""
    request = SystemOneRequest(
        state="anything",
        questions={"q": NoulQuestion(instructions="Is it?")},
        options={"include_evidence": True},
    )
    answer = Engine(_Silent()).answer(request).answers["q"]
    assert answer.evidence == [] and answer.evidence_method == "unavailable"


def test_evidence_outside_the_state_is_rejected_on_every_path():
    request = SystemOneRequest(state="short", questions={"q": NoulQuestion(instructions="Is it?")})
    compiled = compile_request(request)
    for bad in ((0, 99, 0.5), (3, 3, 0.5), (0, 2, 1.5), (0, 2, float("nan"))):
        output = BackendOutput(
            outputs={
                "q": QuestionOutput(question_id="q", kind="noul", logits=(0.0,), evidence=(bad,))
            },
            model_version="x",
        )
        with pytest.raises(ValueError, match="evidence"):
            validate_output(output, compiled)


def test_the_gateway_serves_evidence_and_leaves_the_default_alone():
    client = TestClient(build_app(ServerConfig(backend="lexical")))
    body = {
        "state": "the card transaction was refused at the till",
        "questions": {
            "intent": {
                "type": "choice",
                "instructions": "Route this ticket.",
                "options": [
                    {"name": "card_declined", "criteria": "a card transaction was refused"},
                    {"name": "lost_luggage", "criteria": "baggage missing after a flight"},
                ],
            }
        },
    }
    plain = client.post("/v1/systemone", json=body).json()
    assert "evidence" not in plain["answers"]["intent"]
    explained = client.post(
        "/v1/systemone", json={**body, "options": {"include_evidence": True}}
    ).json()
    answer = explained["answers"]["intent"]
    assert answer["evidence_method"] == "lexical_overlap"
    # Adjacent matches merge: one span, not two.
    assert "card transaction" in {span["text"] for span in answer["evidence"]}
    assert answer["probabilities"] == plain["answers"]["intent"]["probabilities"]


# -- the merge rule -----------------------------------------------------------


def test_kept_tokens_separated_only_by_whitespace_merge_into_one_span():
    text = "a very bad word here"
    tokens = [(0, 1, 0.1), (1, 6, 0.9), (6, 10, 0.7), (10, 15, 0.6), (15, 20, 0.2)]
    assert merge_spans(text, tokens) == [(2, 15, 0.9)]


def test_a_dropped_token_between_two_kept_ones_splits_them():
    text = "bad and bad"
    assert merge_spans(text, [(0, 3, 0.8), (3, 7, 0.1), (7, 11, 0.6)]) == [
        (0, 3, 0.8),
        (8, 11, 0.6),
    ]


def test_a_leading_space_is_never_evidence_and_whitespace_alone_is_dropped():
    text = "x  \n y"
    assert merge_spans(text, [(1, 5, 0.9)]) == []
    assert merge_spans(text, [(4, 6, 0.9)]) == [(5, 6, 0.9)]


def test_two_tokens_that_split_one_character_merge():
    text = "🎯!"
    assert merge_spans(text, [(0, 1, 0.9), (0, 1, 0.7), (1, 2, 0.1)]) == [(0, 1, 0.9)]


def test_a_subword_token_is_widened_to_its_word():
    """The spike's first HateXplain answer returned "kes" out of "likes"."""
    text = "nobody likes you, ok"
    assert merge_spans(text, [(7, 9, 0.2), (9, 12, 0.7)]) == [(7, 12, 0.7)]
    # Punctuation is not a word character: a kept comma does not swallow "you".
    assert merge_spans(text, [(16, 17, 0.9)]) == [(16, 17, 0.9)]


def test_a_run_too_long_to_be_a_word_keeps_the_tokens_own_span():
    text = "x" * 40 + " end"
    assert merge_spans(text, [(10, 12, 0.9)]) == [(10, 12, 0.9)]


def test_token_labels_trim_the_leading_space_before_testing_overlap():
    text = "a bad word"
    offsets = [(0, 1), (1, 5), (5, 10)]
    assert token_labels(text, offsets, [(2, 5)]) == [0, 1, 0]
    assert token_labels(text, [(1, 2)], [(0, 10)]) == [0]


# -- offsets, exact against the text ------------------------------------------


@pytest.mark.parametrize("text", SAMPLES)
def test_bpe_offsets_are_its_own_ids_and_cover_exactly_its_bytes(text):
    tokenizer = BPETokenizer.load()
    triples = tokenizer.encode_with_offsets(text)
    assert [t[0] for t in triples] == tokenizer.encode(text)
    decoder = tokenizer._byte_decoder
    rebuilt = b""
    for token_id, start, end in triples:
        piece = bytes(decoder[c] for c in tokenizer.decoder[token_id])
        # A token's bytes lie inside the characters it claims, and it claims
        # no character it has no byte of.
        covered = text[start:end].encode("utf-8")
        assert piece in covered
        assert len(covered) - len(piece) < 4 * 2
        rebuilt += piece
    assert rebuilt == text.encode("utf-8")
    starts = [t[1] for t in triples]
    assert starts == sorted(starts)


@pytest.mark.parametrize("text", SAMPLES)
def test_hashing_offsets_slice_out_each_piece(text):
    tokenizer = HashingTokenizer()
    triples = tokenizer.encode_with_offsets(text)
    assert [t[0] for t in triples] == tokenizer.encode(text)
    for token_id, start, end in triples:
        assert tokenizer._id(text[start:end]) == token_id


# -- plausibility ---------------------------------------------------------------


def test_token_f1_and_iou_f1_on_a_worked_example():
    gold = [False, True, True, False, True]
    predicted = [False, True, False, False, True]
    precision, recall, f1 = token_prf(predicted, gold)
    assert (precision, recall) == (1.0, 2 / 3)
    assert f1 == pytest.approx(0.8)
    # Runs: predicted [1,2) and [4,5); gold [1,3) and [4,5). [1,2) against
    # [1,3) is IOU 0.5, which matches; [4,5) matches exactly.
    assert iou_f1(predicted, gold) == 1.0
    assert iou_f1([True, True, True, True, True], gold) == 0.0
    assert iou_f1([False] * 5, gold) == 0.0


def test_a_word_is_highlighted_when_any_span_touches_it():
    text = "one two three"
    units = words(text)
    assert highlighted_words(units, [(5, 6)]) == [False, True, False]
    assert highlighted_words(units, [(3, 4)]) == [False, False, False]


def _rationale_case(i: int, state: str, marked: str | None) -> Case:
    rationale = None
    if marked is not None:
        at = state.index(marked)
        rationale = ((at, at + len(marked)),)
    return Case(
        case_id=f"r/{i}",
        request=SystemOneRequest(
            state=state, questions={"toxic": NoulQuestion(instructions="Is this toxic?")}
        ),
        expected={"toxic": Expectation(probability=1.0, rationale=rationale)},
    )


def test_the_lexicon_floor_learns_the_words_people_mark_and_nothing_else():
    training = [_rationale_case(i, f"you are a zorp number {i}", "zorp") for i in range(5)]
    lexicon = RationaleLexicon().fit(training)
    assert lexicon.vocabulary == {"zorp"}
    assert lexicon.spans("a Zorp, again") == [(2, 7)]


def test_only_questions_with_a_non_empty_rationale_are_scored():
    cases = [
        _rationale_case(0, "a zorp here", "zorp"),
        _rationale_case(1, "nothing marked", None),
        dataclasses.replace(
            _rationale_case(2, "empty", None),
            expected={"toxic": Expectation(probability=0.0, rationale=())},
        ),
    ]
    assert [c.case_id for c, _ in rationale_cases(cases)] == ["r/0"]


def test_the_suite_puts_every_floor_beside_the_model():
    training = [_rationale_case(i, f"you are a zorp number {i}", "zorp") for i in range(5)]
    evaluation = [_rationale_case(i, f"what a zorp said {i} times", "zorp") for i in range(3)]
    rows = run_rationale_suite({"model": Engine(LexicalBackend())}, evaluation, training)
    names = [r.name for r in rows]
    assert names == ["model", "lexical floor", "rationale lexicon", "every word"]
    by_name = {r.name: r for r in rows}
    assert by_name["rationale lexicon"].token_f1 == 1.0
    assert by_name["every word"].token_recall == 1.0
    assert by_name["model"].method == "lexical_overlap"
