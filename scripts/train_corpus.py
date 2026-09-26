"""Train and calibrate on a real corpus, and publish the calibration per corpus.

Every calibration number this project has published came from synthetic data:
generated state, decidable predicates, no subjectivity. That exercises the
machinery honestly and it cannot tell you whether the machinery transfers,
which is Stage 2 of `docs/plan.md` and the reason this script exists.

`trigon train` is not reused as-is on purpose. Its report computes a
Bayes-optimal loss, which it can do because it knows the generator and its
noise rate; on a real corpus there is no such number, and a report that
printed one would be inventing a floor. What is reused is everything below
that: the trainer, the per-primitive calibrator selection, the suite, and the
gates.

**The baseline is what makes the number mean anything.** On Banking77 the
marginal predictor scores about 1.3%, so `accuracy_over_baseline` is a real
test here in a way it is not on a four-option synthetic question. A model that
ignores its input cannot pass it by accident.

Splits are disjoint by construction and the calibration split is carved out of
*train*, never out of the evaluation split. Fitting a calibrator on the data
the gates then read is how a calibration number stops meaning anything, and
this repo has made that mistake once already.

    python scripts/train_corpus.py banking77 --out reports/banking77/run.md \
        -n 4000 --epochs 4

**A corpus with human rationales (HateXplain) also trains the evidence head and
publishes its plausibility** -- token F1 and IOU F1 against what the annotators
highlighted -- in the same report, beside the three floors
`trigon.evals.rationale` computes: the lexical floor's evidence, a word list
fitted to the training split's highlights, and highlighting every word. The
same weights are also scored under every attribution they do not serve --
gradient x input and integrated gradients -- so the table shows what the
supervision bought and which unsupervised attribution is the better one.

**Scoring attributions needs no retraining.** ``--weights`` skips training and
re-gates an existing checkpoint, and the plausibility table comes with it:

    python scripts/train_corpus.py hatexplain --weights run.pt -n 0 --seed 0 \\
        --out reports/hatexplain/rescored-seed0.md

**Faithfulness comes with it** (`trigon.evals.faithfulness`): for the first
``--faithfulness-n`` rationale cases, comprehensiveness and sufficiency of
every method the weights can serve, beside a random control and the rationale
lexicon, each probe a real re-ask of a shorter post through the engine.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from trigon.cli import _fit_calibration  # noqa: E402
from trigon.engine import Engine  # noqa: E402
from trigon.evals import (  # noqa: E402
    blocking,
    check_gates,
    render_json,
    render_markdown,
    run_calibration_suite,
)
from trigon.evals.corpora import corpus, load  # noqa: E402
from trigon.limits import MIN_CALIBRATION_SAMPLES  # noqa: E402
from trigon.schema import OptionScoring  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", help="a corpus name from trigon.evals.corpora")
    parser.add_argument("--out", default=None, help="write the report here")
    parser.add_argument("--save-model", default=None)
    parser.add_argument(
        "--weights",
        default=None,
        help=(
            "re-calibrate and re-gate an existing checkpoint instead of training. "
            "The corpus analogue of scripts/regate.py: the splits are derived from "
            "--seed and the corpus, so a re-run with the same flags reads the same "
            "data the training run held out"
        ),
    )
    parser.add_argument("-n", type=int, default=4000, help="training cases (0 = all)")
    parser.add_argument("--calibration-n", type=int, default=1000)
    parser.add_argument(
        "--eval-n",
        type=int,
        default=0,
        help=(
            "evaluation cases; 0 (the default) means as many as "
            "MIN_CALIBRATION_SAMPLES requires, supplementing from unseen train "
            "rows when the corpus's own test split is smaller than the floor"
        ),
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--floor-trials", type=int, default=200)
    parser.add_argument("--option-scoring", default="auto")
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help=(
            "torch intra-op threads. One by default so several seeds in parallel "
            "are cores rather than oversubscription, and so the latency the report "
            "publishes is measured under a known thread count"
        ),
    )
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--backbone",
        default=None,
        help=(
            "a pinned pretrained backbone from trigon.backends.hub (e.g. qwen2.5-1.5b) "
            "instead of the spike; --d-model and --layers are then the backbone's"
        ),
    )
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument(
        "--rationale-weight",
        type=float,
        default=1.0,
        help=(
            "weight of the evidence head's loss on cases carrying a human rationale; "
            "0 trains no head, and the checkpoint serves gradient x input"
        ),
    )
    parser.add_argument(
        "--rationale-eval-n",
        type=int,
        default=0,
        help="score plausibility on at most this many evaluation rationales (0 = all)",
    )
    parser.add_argument(
        "--ig-steps",
        type=int,
        default=0,
        help="integrated gradients' points on the path (0 = the backend's default)",
    )
    parser.add_argument(
        "--ig-chunk",
        type=int,
        default=0,
        help="integrated gradients' points per forward pass (0 = the backend's default)",
    )
    parser.add_argument(
        "--ig-completeness-n",
        type=int,
        default=200,
        help="check integrated gradients' completeness on this many rationale cases",
    )
    parser.add_argument(
        "--faithfulness-n",
        type=int,
        default=500,
        help=(
            "score comprehensiveness and sufficiency on this many rationale cases "
            "(0 = skip). Each case costs up to ten re-asks per method, deduplicated"
        ),
    )
    parser.add_argument(
        "--faithfulness-batch",
        type=int,
        default=16,
        help="re-asks per batched forward in the faithfulness suite (1 = one at a time)",
    )
    parser.add_argument(
        "--hard-labels",
        action="store_true",
        help="train on each annotator distribution's majority vote (the ablation)",
    )
    parser.add_argument(
        "--resume-path",
        default=None,
        help="write a resume file each epoch and continue from it if present",
    )
    parser.add_argument(
        "--max-batch-cells",
        type=int,
        default=0,
        help=(
            "cap batch x sequence^2 per forward pass, splitting a step into "
            "sub-batches that accumulate into the same update; 0 is no cap. "
            "A GPU needs one on long-tailed corpora -- see TrainingConfig"
        ),
    )
    parser.add_argument(
        "--bucket-window",
        type=int,
        default=8,
        help=(
            "sort each window of accumulate x this many shuffled cases by length "
            "before cutting chunks; 1 turns bucketing off (docs/next.md A.5)"
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "cpu, cuda, or auto (cuda when present). Naming cuda on a machine "
            "without one is an error, not a fallback: a run that asked for a GPU "
            "and quietly trained on the CPU publishes a report about hardware it "
            "never used"
        ),
    )
    return parser.parse_args(argv)


def resolve_device(requested: str) -> tuple[str, str]:
    """The device to train on, and the name of the hardware behind it."""
    import torch

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit(f"--device {requested} was asked for and no CUDA device is present")
        return requested, torch.cuda.get_device_name(torch.device(requested))
    import platform

    return requested, platform.processor() or platform.machine()


def splits(spec, args) -> tuple[list, list, list, int]:
    """Train and calibration out of the train split, evaluation out of test.

    The calibration split is carved from train rather than from test for the
    reason the whole repo is built around: a calibrator fitted on the data the
    gates read reports its own fit, not the model's calibration. Shuffled with
    a seeded RNG so the carve is reproducible and so it is not the corpus's own
    ordering, which on several of these is grouped by label.

    **The evaluation split is topped up from train when the corpus's own test
    split is below `MIN_CALIBRATION_SAMPLES`, and this is the interesting
    part.** Banking77's test split is 3,080 rows and the floor is 5,000, so
    the first run of this script failed `sample_size` and `gate_is_testable`
    on every seed: a perfectly calibrated model scores ECE 0.0221 on 1,500
    cases, which is most of the 0.05 gate, so the gate was not a test. That is
    not a threshold to widen -- `CLAUDE.md` is explicit that the run gets
    fixed rather than the gate -- and it is a real defect in the plan's Stage 2
    done-condition, which asks for per-corpus ECE without noticing that a
    corpus can be too small to carry one.

    The top-up rows are disjoint from both training and calibration and were
    never seen by either, so they are held-out data by the only definition
    that matters. They come from the train *distribution* rather than the
    test one, which is a real difference and is why the report states how many
    of each it used instead of printing one number.
    """
    train = load(spec.name, "train", purpose="train")
    evaluation = load(spec.name, "test", purpose="eval")
    # Both splits, and this is not defensive tidying. Banking77's test.csv is
    # ordered by label, so the first version of this took `evaluation[:1000]`
    # and got an evaluation set of one intent: the marginal predictor scored
    # 1.0000 and `accuracy_over_baseline` read -1.0000. A gate cannot catch a
    # subset that is not the distribution, because nothing about the number
    # looks wrong -- the model really did lose to that baseline.
    rng = random.Random(f"corpus:{spec.name}:{args.seed}")
    rng.shuffle(train)
    rng.shuffle(evaluation)

    calibration_n = min(args.calibration_n, len(train) // 4)
    calibration = train[:calibration_n]
    remaining = train[calibration_n:]

    if args.eval_n:
        evaluation = evaluation[: args.eval_n]
        topped_up = 0
    else:
        # As many as the floor requires, taken from rows nothing else will see
        # -- but never more than half the pool, because the top-up must not
        # eat the training set. The first version capped it at `len(remaining)
        # - 1` and, on a corpus with a 4,000-row train split, reached the floor
        # by leaving **one** training case. Every count in the report was
        # correct and the report was worthless.
        #
        # When half the pool is not enough to reach the floor, the floor is
        # not reached and `sample_size` fails. That is the right answer: a
        # corpus too small to supply both a trainable set and a floor-sized
        # evaluation is a corpus this gate cannot certify, and cannibalising
        # training to make the gate pass is the same move as widening it.
        wanted = max(0, MIN_CALIBRATION_SAMPLES - len(evaluation))
        topped_up = min(wanted, len(remaining) // 2)
        evaluation = evaluation + remaining[:topped_up]
        remaining = remaining[topped_up:]

    training = remaining if args.n == 0 else remaining[: args.n]
    return training, calibration, evaluation, topped_up


def baseline_accuracy(train: list, evaluation: list) -> dict[str, float]:
    """What a model that ignores its input scores, per question.

    The modal label is computed from the *training* split and applied to
    evaluation, because a baseline fitted on the evaluation split is not a
    baseline, it is an oracle with one degree of freedom.

    Per question rather than pooled. A corpus like HelpSteer2 asks five Score
    questions over the same state and their marginals differ -- pooling them
    into one number would let a well-predicted question carry a badly
    predicted one, which is the same cancellation that made pooled ECE
    misleading here before.
    """
    counts: dict[str, dict[int, int]] = {}
    for case in train:
        for qid, expectation in case.expected.items():
            counts.setdefault(qid, {})
            label = expectation.label
            counts[qid][label] = counts[qid].get(label, 0) + 1
    out: dict[str, float] = {}
    for qid, tally in counts.items():
        modal = max(tally, key=lambda k: tally[k])
        seen = [c for c in evaluation if qid in c.expected]
        hits = sum(1 for c in seen if c.expected[qid].label == modal)
        out[qid] = hits / len(seen) if seen else 0.0
    return out


def majority_labels(cases: list) -> list:
    """The same cases with each annotator distribution replaced by its mode.

    The ablation for `--hard-labels`: training on the majority vote instead of
    the distribution, on identical splits, so the target is the only thing that
    differs. Ties go to the lower rating -- a fixed rule, so two runs agree.
    Only the training split is rewritten; calibration and evaluation keep the
    drawn-annotator outcome both arms are scored against.
    """
    import dataclasses

    from trigon.evals.harness import Expectation

    def mode(expectation):
        d = expectation.distribution
        if d is None:
            return expectation
        return Expectation(label=max(range(len(d)), key=lambda k: (d[k], -k)))

    return [
        dataclasses.replace(c, expected={q: mode(e) for q, e in c.expected.items()}) for c in cases
    ]


def marginal_scores(train: list, evaluation: list) -> tuple[float, float]:
    """Brier and NLL of the model that ignores its input, pooled like the suite's.

    `accuracy_over_baseline` compares argmaxes, and on a corpus whose labels
    are single annotators it cannot be cleared by anything -- an oracle that
    knows the other annotators' ratings of the same response reaches +0.021
    on HelpSteer2 (`reports/helpsteer2/ceiling.md`). A proper scoring rule
    still separates a model that reports each response's spread of opinion
    from one that reports the population's, so the report states what the
    population's scores: the per-question label distribution of the training
    split, add-one smoothed, scored on the evaluation labels. Reported beside
    the gates, not as one.
    """
    from trigon.calibration.metrics import brier, negative_log_likelihood

    probs, labels = [], []
    for qid in train[0].expected:
        counts: dict[int, int] = {}
        for case in train:
            label = case.expected[qid].label
            counts[label] = counts.get(label, 0) + 1
        levels = max(max(counts) + 1, len(counts))
        for case in evaluation:
            levels = max(levels, case.expected[qid].label + 1)
        total = sum(counts.values()) + levels
        marginal = [(counts.get(k, 0) + 1) / total for k in range(levels)]
        for case in evaluation:
            probs.append(marginal)
            labels.append(case.expected[qid].label)
    return brier(probs, labels), negative_log_likelihood(probs, labels)


def header(
    spec, args, train, calibration, evaluation, marginal, topped_up: int, hardware: str = ""
) -> str:
    questions = train[0].request.questions
    first = next(iter(questions.values()))
    labels = getattr(first, "options", None) or getattr(first, "levels", [])
    return "\n".join(
        [
            f"# {spec.name}",
            "",
            f"**{spec.attribution}**",
            "",
            "Real labelled data, not the synthetic generator. There is no",
            "Bayes-optimal loss to quote here -- the labels are human and the",
            "corpus does not come with a noise rate -- so the floor reported",
            "below is the marginal predictor, which is the floor that matters",
            "for the `accuracy_over_baseline` gate.",
            "",
            "| | |",
            "| --- | ---: |",
            f"| Training cases | {len(train):,} |",
            f"| Calibration cases (held out of train) | {len(calibration):,} |",
            f"| Evaluation cases | {len(evaluation):,} |",
            f"| — from the corpus's own test split | {len(evaluation) - topped_up:,} |",
            f"| — held out of train to reach the floor | {topped_up:,} |",
            f"| Questions per request | {len(questions)} |",
            f"| Labels per question | {len(labels)} |",
            "",
            "| Question | Marginal predictor |",
            "| --- | ---: |",
            *(f"| `{qid}` | {value:.4f} |" for qid, value in sorted(marginal.items())),
            "",
            "| Marginal predictor, pooled | Brier | NLL |",
            "| --- | ---: | ---: |",
            "| {} | {:.4f} | {:.4f} |".format(
                "ignores its input", *marginal_scores(train, evaluation)
            ),
            "",
            "A proper scoring rule, where argmax accuracy cannot separate a model",
            "from the population: compare the model's Brier in the Suites table.",
            "",
            "| | |",
            "| --- | ---: |",
            f"| Epochs | {args.epochs} |",
            f"| Seed | {args.seed} |",
            *([f"| Device | {hardware} |"] if hardware else []),
            (
                f"| Model | {args.backbone}, LoRA rank {args.lora_rank} |"
                if args.backbone
                else f"| Model | reference spike, d_model {args.d_model}, {args.layers} layers |"
            ),
            "",
            "```",
            f"python scripts/train_corpus.py {spec.name} -n {args.n} "
            f"--calibration-n {args.calibration_n} --eval-n {args.eval_n} "
            f"--epochs {args.epochs} --lr {args.lr} --accumulate {args.accumulate} "
            f"--d-model {args.d_model} --layers {args.layers} --seed {args.seed} "
            f"--device {args.device}"
            + (f" --backbone {args.backbone} --lora-rank {args.lora_rank}" if args.backbone else "")
            + (f" --max-batch-cells {args.max_batch_cells}" if args.max_batch_cells else "")
            + (" --hard-labels" if args.hard_labels else "")
            + (
                f" --rationale-weight {args.rationale_weight}"
                if args.rationale_weight != 1.0
                else ""
            ),
            "```",
            "",
            *(
                [
                    f"`MIN_CALIBRATION_SAMPLES` is {MIN_CALIBRATION_SAMPLES:,} and this",
                    "corpus's test split is smaller, so the evaluation set is topped up",
                    "from rows held out of train that neither training nor calibration",
                    "saw. They are held-out data by the only definition that matters and",
                    "they come from the train distribution, which is why the split is",
                    "reported above rather than summed into one number.",
                    "",
                ]
                if topped_up
                else []
            ),
            "**One seed is one sample from a distribution nobody measured.**",
            "See `CLAUDE.md`: certify on the median and the spread, never on a",
            "single draw. This report is a measurement, not a certification.",
            "",
        ]
    )


def evidence_section(backend, engine, train, evaluation, args):
    """The plausibility table, or None for a corpus without human rationales.

    Scored on the evaluation split only, and the rationale lexicon floor is
    fitted on the training split only: fitted on the cases it is scored on, a
    word list is an oracle with a vocabulary.
    """
    from trigon.evals.rationale import rationale_cases, render_plausibility, run_rationale_suite

    scored = [case for case in evaluation if rationale_cases([case])]
    if not scored:
        return None
    if args.rationale_eval_n:
        scored = scored[: args.rationale_eval_n]
    from trigon.backends.torch_readout import UNSUPERVISED_EVIDENCE_METHODS
    from trigon.evals.rationale import score_engine

    engines = {"model": engine}
    rows = run_rationale_suite(engines, scored, train)
    # Same weights, every attribution the served row is not: beside a trained
    # head, what the supervision bought; beside gradient x input, whether
    # integrated gradients is the better unsupervised method -- the measurement
    # `docs/decisions.md` makes the default wait on. An existing checkpoint
    # gets every row through --weights, with no retraining.
    served = backend._resolved_evidence_mode()
    others = [m for m in UNSUPERVISED_EVIDENCE_METHODS if m != served]
    for position, method in enumerate(others, start=1):
        backend.evidence_mode = method
        try:
            label = f"model, {method.replace('_', ' ')}"
            rows.insert(position, score_engine(label, engine, scored))
        finally:
            backend.evidence_mode = "auto"
    completeness = ig_completeness(backend, engine, scored, args.ig_completeness_n)
    faithful = (
        faithfulness_section(backend, engine, scored, train, args) if args.faithfulness_n else None
    )
    text = "\n".join(
        [
            "",
            "## Evidence: plausibility against human rationales",
            "",
            "What the model highlights, scored against what the annotators",
            "highlighted, on words of the state. Token F1 is per-case F1 over",
            "highlighted words, averaged; IOU F1 counts a predicted span as found",
            "when it overlaps a human span by at least half their union. The three",
            "rows after the model are floors, and the rationale lexicon is the one",
            "that matters: every word highlighted in at least half its training",
            "occurrences. A highlighter that does not beat it has learned a",
            "vocabulary, not a reading. The model rows are one set of weights",
            "under each evidence method; `model` is the one it serves.",
            "",
            render_plausibility(rows),
            "",
            *render_completeness(completeness, backend),
            *(faithful[0] if faithful else []),
        ]
    )
    return text, rows, completeness, faithful


def faithfulness_section(backend, engine, scored, train, args):
    """Comprehensiveness and sufficiency for every method the weights can serve.

    `trigon.evals.faithfulness` has the definitions and the choices. The span
    head is scored only on a checkpoint trained on rationales: an untrained
    head is a random projection, and the random control already measures one.
    """
    import time

    from trigon.backends.torch_readout import UNSUPERVISED_EVIDENCE_METHODS
    from trigon.evals.faithfulness import (
        FAITHFULNESS_BINS,
        engine_scorer,
        render_faithfulness,
        run_faithfulness_suite,
    )

    methods = ["span_head"] if backend.config.evidence_supervised else []
    methods += list(UNSUPERVISED_EVIDENCE_METHODS)
    inner = engine_scorer(engine)

    def under(method):
        def score(case, qid, text):
            backend.evidence_mode = method
            try:
                return inner(case, qid, text)
            finally:
                backend.evidence_mode = "auto"

        return score

    scorers = {f"model, {m.replace('_', ' ')}": under(m) for m in methods}
    started = time.perf_counter()
    rows, prober = run_faithfulness_suite(
        scorers,
        engine,
        scored,
        train,
        n=args.faithfulness_n,
        batch_size=args.faithfulness_batch,
        seed=args.seed,
    )
    cost = {
        "cases": rows[0].n,
        "methods": len(rows),
        "forwards": prober.forwards,
        "forward_seconds": round(prober.seconds, 1),
        "total_seconds": round(time.perf_counter() - started, 1),
        "batch": args.faithfulness_batch,
    }
    bins = ", ".join(f"{b:.0%}" for b in FAITHFULNESS_BINS)
    lines = [
        "## Evidence: faithfulness",
        "",
        "Did the model use what it highlights? For the label it selects on the",
        f"full post, over the first {rows[0].n:,} of these cases: **comprehensiveness**",
        "is how far that label's probability falls when the top-k% of words by",
        "the method's own scores are deleted from the post and the post is asked",
        "again (higher: the words mattered); **sufficiency** is how far it falls",
        "when only those words are kept (lower: they suffice). ERASER's AOPC, the",
        f"mean over k in {bins}, on the model's uncalibrated distribution.",
        "`random` and the `rationale lexicon` are controls scored the same way:",
        "a method that does not beat `random` found nothing the answer needed,",
        "and one that does not beat the lexicon found no more than vocabulary.",
        "The `− random` and `− lexicon` columns are the paired per-case",
        "difference from each control, with a 95% interval.",
        "",
        render_faithfulness(rows),
        "",
        f"Cost: {cost['forwards']:,} re-asks in {cost['forward_seconds']:.1f} s "
        f"(batches of {cost['batch']}), {cost['total_seconds']:.1f} s with attribution.",
        "",
    ]
    return lines, rows, cost


def ig_completeness(backend, engine, cases, limit: int) -> dict | None:
    """How far integrated gradients' attributions sum from what they must.

    Completeness -- the attributions over the state summing to the selected
    label's log-probability at the input minus at the zero baseline -- is the
    property that defines the method, and the quadrature only approximates
    it. `tests/test_independence.py` checks it on one request of an untrained
    spike; this measures it on the weights and the data the plausibility rows
    come from, because a pre-norm backbone concentrates the path's change near
    the baseline and a step count that suffices on one model need not on
    another. Relative error over cases whose difference is at least 0.01 nats;
    smaller differences make any relative error meaningless.
    """
    import statistics

    from trigon.evals.rationale import rationale_cases

    pairs = rationale_cases(cases)[: max(0, limit)]
    if not pairs:
        return None
    errors, deltas = [], []
    was_training = backend.model.training
    backend.model.eval()
    try:
        for case, qid in pairs:
            request = case.request
            compiled = engine.compiler.compile_request(request)
            attributions = backend.integrated_gradients(compiled, request)
            delta = backend.path_difference(compiled, request)[qid]
            deltas.append(abs(delta))
            if abs(delta) >= 0.01:
                errors.append(abs(float(attributions[qid].sum()) - delta) / abs(delta))
    finally:
        backend.model.train(was_training)
    ordered = sorted(errors)

    def quantile(q: float) -> float | None:
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))] if ordered else None

    return {
        "cases": len(pairs),
        "scored": len(errors),
        "median_abs_delta_nats": statistics.median(deltas),
        "median_relative_error": quantile(0.5),
        "p90_relative_error": quantile(0.9),
        "max_relative_error": ordered[-1] if ordered else None,
    }


def render_completeness(result: dict | None, backend) -> list[str]:
    if result is None or result["scored"] == 0:
        return []
    return [
        "Integrated gradients' completeness on the first "
        f"{result['cases']:,} of these cases ({backend.ig_steps} points, "
        f"`u ** {backend.ig_power}` spacing): relative error of the summed attributions "
        "against the log-probability difference they must add up to, median "
        f"{result['median_relative_error']:.4f}, p90 {result['p90_relative_error']:.4f}, "
        f"max {result['max_relative_error']:.4f}, over the {result['scored']:,} whose "
        f"difference is at least 0.01 nats (median difference "
        f"{result['median_abs_delta_nats']:.3f}). A large error means too few points, "
        "and the `integrated gradients` row is then a quadrature artefact, not the method.",
        "",
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Set here rather than inherited from the trainer, which is where it used
    # to come from: `TrainingConfig.torch_threads` is 1 and `run_training`
    # applies it, so every eval after a training run was single-threaded by
    # accident. Re-gating a checkpoint skips training, so four seeds in
    # parallel each took four threads on four cores -- 3.5x slower, and worse
    # than slow, because the suite publishes p50 and p99 latency and those
    # numbers are meaningless under oversubscription. The accuracy and the ECE
    # would have been right and the latency quietly wrong.
    import torch

    from trigon.backends.torch_readout import ReadoutConfig, TorchReadoutBackend
    from trigon.training import TrainingConfig
    from trigon.training import train as run_training

    torch.set_num_threads(args.torch_threads)
    device, hardware = resolve_device(args.device)

    spec = corpus(args.corpus)
    if not spec.permits("train"):
        # Before a model is built: the loader would refuse anyway, but a
        # traceback after a 1.5B backbone has loaded reads like a crash rather
        # than the licence policy doing its job. Re-gating counts too -- the
        # calibrator is fitted on this corpus and ships with the weights.
        print(
            f"{spec.name} is {spec.tier} ({spec.licence}) and may not be trained or "
            "calibrated on; it evaluates only (docs/decisions.md section 3, Q17)",
            file=sys.stderr,
        )
        return 2
    if args.weights:
        backend = TorchReadoutBackend.load(args.weights)
    elif args.backbone:
        from trigon.backends.qwen_readout import QwenReadoutBackend

        # Loaded straight onto the device: a 1.5B model materialised on the
        # CPU first and moved costs a second copy of it in a container's RAM.
        backend = QwenReadoutBackend.from_backbone(
            args.backbone, device=device, seed=args.seed, lora_rank=args.lora_rank
        )
        backend.model.checkpointing = True
    else:
        backend = TorchReadoutBackend(
            config=ReadoutConfig(d_model=args.d_model, n_layers=args.layers), seed=args.seed
        )
    backend.to(device)
    backend.ig_steps = args.ig_steps or backend.ig_steps
    backend.ig_chunk = args.ig_chunk or backend.ig_chunk
    hardware = f"{device} ({hardware})"
    print(f"{args.corpus}: training on {hardware}", file=sys.stderr)
    compiler = backend.make_compiler(option_scoring=OptionScoring(args.option_scoring))
    train, calibration, evaluation, topped_up = splits(spec, args)
    marginal = baseline_accuracy(train, evaluation)
    summary = " ".join(f"{qid}={value:.4f}" for qid, value in sorted(marginal.items()))
    print(
        f"{spec.name}: {len(train)} train, {len(calibration)} calibration, "
        f"{len(evaluation)} eval; marginals {summary}",
        file=sys.stderr,
    )
    report = None
    if not args.weights:
        report = run_training(
            backend,
            majority_labels(train) if args.hard_labels else train,
            TrainingConfig(
                epochs=args.epochs,
                learning_rate=args.lr,
                accumulate=args.accumulate,
                seed=args.seed,
                log_every=args.log_every,
                validation_fraction=args.validation_fraction,
                max_batch_cells=args.max_batch_cells or None,
                resume_path=args.resume_path,
                bucket_window=args.bucket_window,
                rationale_weight=args.rationale_weight,
            ),
            compiler=compiler,
        )
    if device.startswith("cuda"):
        # Sizing the GPU for the next run needs this, and nothing else records it.
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"{args.corpus}: peak GPU memory in training {peak:.1f} GiB", file=sys.stderr)
    # **Evaluate the path we serve.** The gateway defaults the schema prefix
    # cache on (reports/cache/README.md: 6x at 77 options), so a report
    # measured with it off describes a deployment nobody runs. Safe here for
    # the reason it is unsafe in training: the weights are fixed now, and
    # `_prefix_for` refuses to cache while the model is in training mode
    # anyway. It moves answers by ~5e-08, which is the same cross-shape
    # float32 rounding already in the ledger and below any gate.
    backend.cache_prefixes = True

    engine = Engine(backend, compiler=compiler)
    before, _ = run_calibration_suite(
        engine, evaluation, suite=f"{spec.name}/uncalibrated", floor_trials=args.floor_trials
    )
    scaler, isotonic = _fit_calibration(engine, calibration)
    calibrated = Engine(backend, compiler=compiler, scaler=scaler, isotonic=isotonic)
    after, slices = run_calibration_suite(
        calibrated, evaluation, suite=f"{spec.name}/calibrated", floor_trials=args.floor_trials
    )
    # Scored against one annotator drawn per case: argmax accuracy is capped
    # below its gate for every predictor, so the proper score carries the
    # "uses its input" term instead (limits.MIN_BRIER_SKILL_OVER_MARGINAL).
    gates = check_gates(
        after,
        slices=slices,
        marginal_brier=marginal_scores(train, evaluation)[0] if spec.annotator_lists else None,
        # Q16: blocking once a real backbone lands, which a backbone run is.
        require_per_question=bool(args.backbone),
    )
    markdown = header(
        spec, args, train, calibration, evaluation, marginal, topped_up, hardware
    ) + render_markdown([before, after], gates, slices, gated=after)
    plausibility = evidence_section(backend, calibrated, train, evaluation, args)
    if plausibility is not None:
        markdown += plausibility[0]
    print(markdown)

    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown)
        payload = json.loads(render_json([before, after], gates, slices, gated=after))
        if plausibility is not None:
            payload["plausibility"] = [row.to_dict() for row in plausibility[1]]
            payload["ig_completeness"] = plausibility[2]
            if plausibility[3] is not None:
                payload["faithfulness"] = [row.to_dict() for row in plausibility[3][1]]
                payload["faithfulness_cost"] = plausibility[3][2]
        out.with_suffix(".json").write_text(json.dumps(payload, indent=2))
        stem = out.with_suffix("")
        scaler.save(pathlib.Path(f"{stem}-temperatures.json"))
        if isotonic.knots:
            isotonic.save(pathlib.Path(f"{stem}-isotonic.json"))
        if report is not None:
            pathlib.Path(f"{stem}-training.json").write_text(json.dumps(report.to_dict(), indent=2))
        if args.save_model:
            backend.save(pathlib.Path(args.save_model))
        print(f"\nwrote {out}", file=sys.stderr)
    return 0 if all(g.passed for g in blocking(gates)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
