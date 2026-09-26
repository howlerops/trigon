#!/usr/bin/env python
"""What the schema KV cache actually saves, in milliseconds.

    python scripts/cache_bench.py --weights reports/banking77/b77-seed3.pt

`docs/next.md` B.3. The cache has been built since the KV-prefix commit and
its saving has never been timed: the one benchmark that ran did so under four
concurrent training jobs, which is not a measurement. Until there is a number,
"the schema is cacheable" is an architectural property with no time attached,
and that is the whole reason the gateway defaults it off.

**What makes this measurable at all is that the schema is most of the
request.** On a Banking77 request the 77 option names compile to 606 of 609
tokens. A gateway answering the same schema over and over — which is what a
classifier deployment *is* — recomputes that block on every call.

Three things this script is careful about, because the last cost measurement
in this repository got two of them wrong:

1. **The thread count is set and reported.** `scripts/train_corpus.py` learned
   this the hard way: torch defaults to one thread per core, and an unnamed
   thread count makes a latency number unreproducible.
2. **The cache is warmed before it is timed.** The first request on a schema
   fills the prefix; timing it measures a miss and calls it a hit.
3. **Both arms answer identically.** A cache that returns different numbers is
   not faster, it is broken, so the script checks agreement before it reports
   a saving and refuses to print one if they disagree.
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from trigon.engine import Engine  # noqa: E402
from trigon.types import ChoiceQuestion, DecisionRequest, NoulQuestion  # noqa: E402

# A classifier deployment's shape: one large option set, asked repeatedly.
STATES = [
    "my card payment was declined at the till and I do not know why",
    "where is the refund you promised me three weeks ago",
    "the app will not let me log in after I changed my phone",
    "I was charged twice for the same transaction this morning",
    "can I use my card abroad without paying an extra fee",
]


def _request(state: str, options: int, extra_questions: int) -> DecisionRequest:
    questions: dict = {
        "intent": ChoiceQuestion(
            instructions="Which banking intent does this customer message express?",
            options=[{"name": f"intent number {i}"} for i in range(options)],
        )
    }
    for i in range(extra_questions):
        questions[f"flag_{i}"] = NoulQuestion(instructions=f"Is condition {i} present?")
    return DecisionRequest(state=state, questions=questions)


def _time(engine: Engine, requests: list[DecisionRequest], repeats: int) -> list[float]:
    samples = []
    for _ in range(repeats):
        for request in requests:
            start = time.perf_counter()
            engine.answer(request)
            samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=None, help="a checkpoint; random weights otherwise")
    parser.add_argument("--options", type=int, default=77, help="option count for the Choice")
    parser.add_argument("--extra-questions", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args(argv)

    import torch

    from trigon.backends.torch_readout import TorchReadoutBackend

    torch.set_num_threads(args.torch_threads)

    requests = [_request(s, args.options, args.extra_questions) for s in STATES]
    arms = {}
    answers = {}
    for name, cache in (("cache off", False), ("cache on", True)):
        backend = (
            TorchReadoutBackend.load(args.weights) if args.weights else TorchReadoutBackend(seed=0)
        )
        backend.cache_prefixes = cache
        engine = Engine(backend, compiler=backend.make_compiler())
        # Warm: the first call on a schema fills the prefix, and timing a miss
        # as though it were a hit is how a cache benchmark flatters itself.
        _time(engine, requests, args.warmup)
        arms[name] = _time(engine, requests, args.repeats)
        answers[name] = [engine.answer(r).answers["intent"].probabilities for r in requests[:1]]

    off, on = arms["cache off"], arms["cache on"]

    # A cache that answers differently is not faster, it is broken.
    drift = max(
        abs(a[k] - b[k])
        for a, b in zip(answers["cache off"], answers["cache on"], strict=True)
        for k in a
    )

    sample = _request(STATES[0], args.options, args.extra_questions)
    plain = TorchReadoutBackend.load(args.weights) if args.weights else TorchReadoutBackend(seed=0)
    usage = Engine(plain, compiler=plain.make_compiler()).answer(sample).usage

    print(f"torch threads      : {torch.get_num_threads()}")
    print(f"options            : {args.options}")
    print(f"extra questions    : {args.extra_questions}")
    print(f"tokens per request : {usage.prefill_tokens}")
    print(f"  of which schema  : {usage.schema_tokens}")
    print(f"samples per arm    : {len(off)}")
    print()
    print(f"{'arm':<12} {'p50 ms':>9} {'p90 ms':>9} {'mean ms':>9}")
    print("-" * 42)
    for name, samples in arms.items():
        ordered = sorted(samples)
        p90 = ordered[int(len(ordered) * 0.9)]
        print(
            f"{name:<12} {statistics.median(samples):>9.2f} {p90:>9.2f} "
            f"{statistics.fmean(samples):>9.2f}"
        )
    print()
    saving = statistics.median(off) - statistics.median(on)
    share = saving / statistics.median(off) * 100 if statistics.median(off) else 0.0
    print(f"saving             : {saving:+.2f} ms ({share:+.1f}%)")
    print(f"answers agree to   : {drift:.3e}")
    if drift > 1e-6:
        print()
        print("REFUSING THE NUMBER: the two arms do not answer the same question.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
