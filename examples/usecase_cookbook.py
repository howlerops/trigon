#!/usr/bin/env python
"""Any committed use case, end to end, on a fresh clone.

    python examples/usecase_cookbook.py sentiment
    python examples/usecase_cookbook.py moderation --weights reports/run.pt
    python examples/usecase_cookbook.py --list

`docs/next.md` C.2. Three use cases are committed in `trigon.usecases` and
priced in `docs/pricing.md`, and until now none had a worked example: a reader
could read what the shape *would* be and not run one.

**One driver rather than three scripts.** The use cases already live in one
place; a cookbook per file would be three copies of the same twenty lines,
diverging the first time one of them was edited. `examples/triage_cookbook.py`
stays separate because it teaches the *concepts* — what a distribution buys
that a string does not — while this one answers "what does my use case
actually return, and what does it cost".

It runs on the lexical floor, so it works with no weights and no GPU. The
answers are then nonsense, which is the point of a floor: it exercises the
whole path — compile, answer, calibrate, route, price — and tells you nothing
about quality. Pass `--weights` from a `trigon train` or
`scripts/train_corpus.py` run for answers worth reading.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from trigon.engine import Engine  # noqa: E402
from trigon.types import DecisionRequest, NoulAnswer  # noqa: E402
from trigon.usecases import all_use_cases, use_case  # noqa: E402

ESCALATE_BELOW = 0.35


def _engine(weights: str | None, backend: str):
    if backend == "torch" or weights:
        from trigon.backends.torch_readout import TorchReadoutBackend

        made = TorchReadoutBackend.load(weights) if weights else TorchReadoutBackend(seed=0)
        return Engine(made, compiler=made.make_compiler())
    from trigon.backends.lexical import LexicalBackend

    return Engine(LexicalBackend())


def _render(qid: str, answer) -> list[str]:
    """One answer, in the terms its primitive actually has.

    A Noul is printed without a confidence because it does not have one: for a
    binary question the probability already is one, and inventing
    `max(p, 1-p)` here would put a number on screen that no model produced.
    """
    lines = [f"  {qid}"]
    if isinstance(answer, NoulAnswer):
        lines.append(f"    probability   {answer.probability:.3f}   (a Noul carries no confidence)")
        return lines
    if answer.type == "score":
        lines.append(f"    score         {answer.score:.3f}")
    else:
        lines.append(f"    selected      {answer.selected}")
    lines.append(f"    confidence    {answer.confidence:.3f}")
    ranked = sorted(answer.probabilities.items(), key=lambda kv: -kv[1])
    top = ", ".join(f"{name} {p:.3f}" for name, p in ranked[:4])
    lines.append(f"    distribution  {top}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("use_case", nargs="?", help="a name from trigon.usecases")
    parser.add_argument("--list", action="store_true", help="show the committed use cases")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--backend", default="lexical", choices=("lexical", "torch"))
    parser.add_argument(
        "--escalate-below",
        type=float,
        default=ESCALATE_BELOW,
        help="confidence under which a Choice or Score is sent to a human",
    )
    args = parser.parse_args(argv)

    if args.list or not args.use_case:
        print("committed use cases:\n")
        for case in all_use_cases():
            print(f"  {case.name:<16} {case.purpose}")
        print("\nrun one:  python examples/usecase_cookbook.py <name>")
        return 0 if args.list else 1

    case = use_case(args.use_case)
    request = DecisionRequest(state=case.state, questions=case.questions)
    engine = _engine(args.weights, args.backend)
    response = engine.answer(request)

    print(f"# {case.name}")
    print()
    print(case.purpose)
    if case.notes:
        print()
        print(case.notes.strip())
    print()
    print("## The state it reads")
    print()
    text = case.state if isinstance(case.state, str) else str(case.state)
    print("  " + text.strip().replace("\n", "\n  ")[:600])
    print()
    print(f"## {len(case.questions)} answers, one forward pass")
    print()
    for qid, answer in response.answers.items():
        print("\n".join(_render(qid, answer)))
    print()

    # Confidence routing: the thing a distribution buys that a string does not.
    # A Noul is excluded rather than thresholded on its probability, because
    # 0.5 is its *most uncertain* point and 0.02 is a confident "no" -- reading
    # a low probability as low confidence would escalate exactly the cases the
    # model is surest about.
    undecided = [
        qid
        for qid, answer in response.answers.items()
        if not isinstance(answer, NoulAnswer) and answer.confidence < args.escalate_below
    ]
    print("## Routing")
    print()
    if undecided:
        print(
            f"  below {args.escalate_below:.2f} confidence, send to a human: {', '.join(undecided)}"
        )
    else:
        print(f"  every answer is at or above {args.escalate_below:.2f} confidence")
    print()

    usage = response.usage
    print("## What it cost")
    print()
    print(f"  prefill tokens        {usage.prefill_tokens:,}")
    print(f"    of which schema     {usage.schema_tokens:,}")
    print(f"    cached this call    {usage.cached_schema_tokens:,}")
    print(f"  readout slots         {usage.readout_tokens:,}")
    print(f"  wall clock            {response.timing.total_ms:.1f} ms")
    print()
    print(f"  at {case.monthly_volume:,} decisions a month, that is")
    print(f"    {usage.prefill_tokens * case.monthly_volume / 1e6:,.0f} MTok of prefill")
    print()
    print("  The schema half is identical on every call, so a gateway serving")
    print("  this use case caches it: reports/cache/README.md measures 6x at")
    print("  77 options. scripts/price.py compares the whole thing against a")
    print("  prompted baseline with exact token counts on both sides.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
