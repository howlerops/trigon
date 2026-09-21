#!/usr/bin/env python
"""What one decision costs, per use case, and where the break-even sits.

    python scripts/price.py
    python scripts/price.py --use-case sentiment --verbose
    python scripts/price.py --typed-usd-per-mtok 0.007 --llm-usd-per-mtok 0.25

**The measured part and the assumed part are kept apart on purpose.** Token
counts on both sides are exact: the typed path is counted by `SchemaCompiler`,
and the prompted-LLM path is counted by building the prompt a caller would
actually send and running it through the same tokenizer. Prices per token are
*not* measured here and are not defaulted to anything that looks sourced --
`docs/roadmap.md` records the one cost figure this project inherited
($0.007/MTok on an L4) as arithmetic over unsourced inputs, pending a burn-in.

So the headline this prints is the **token ratio**, which needs no price at
all, and the **break-even rate**: the $/MTok at which the typed path stops
being cheaper than the prompted one. Both are exact. Money figures appear only
when you supply the rates, and are labelled as yours.

**What the prompted baseline is.** One call, structured output, the same
questions in one go -- the strongest form of the alternative rather than the
weakest. It is not N separate calls, and it is not chain-of-thought. A baseline
chosen to lose is not a baseline.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))


def _prompt_for(use_case) -> tuple[str, str, str]:
    """The prompt a caller would send an LLM, split the way billing splits it.

    Returns the **reusable prefix** (instruction and label sets, identical on
    every request), the **per-request part** (the state), and the reply the
    model has to generate.

    Splitting it matters, and getting it wrong is how this comparison gets
    rigged. The typed path's schema is a cacheable prefix -- that is the
    architectural claim. So is the prompted path's instruction block, on any
    provider with prompt caching. Comparing a cached typed path against an
    uncached prompted one would be scoring the alternative with a feature
    turned off, which the first draft of this script did.

    Deliberately terse: no examples, no chain-of-thought, no role-play. Every
    token added here makes the comparison flatter, so it is kept to what the
    job needs -- the instruction, the label sets, the state, and a JSON reply.
    """
    from trigon.schema.compiler import render_state
    from trigon.types import ChoiceQuestion, NoulQuestion, ScoreQuestion

    lines = ["Answer each question about the record. Reply with JSON only."]
    reply: dict[str, object] = {}
    for qid, question in use_case.questions.items():
        if isinstance(question, ChoiceQuestion):
            names = [o.name for o in question.options]
            detail = "; ".join(f"{o.name}: {o.criteria}" for o in question.options if o.criteria)
            lines.append(f"{qid}: {question.instructions} One of: {', '.join(names)}.")
            if detail:
                lines.append(f"  ({detail})")
            reply[qid] = names[0]
        elif isinstance(question, ScoreQuestion):
            names = [level.name for level in question.levels]
            lines.append(f"{qid}: {question.instructions} One of: {', '.join(names)}.")
            reply[qid] = names[0]
        elif isinstance(question, NoulQuestion):
            lines.append(f"{qid}: {question.instructions} yes or no.")
            reply[qid] = "yes"
    return (
        "\n".join(lines),
        render_state(use_case.state),
        json.dumps(reply, separators=(",", ":")),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--use-case", default=None, help="one by name; default is all")
    parser.add_argument(
        "--typed-usd-per-mtok",
        type=float,
        default=None,
        help="your rate for the typed model; unset prints token ratios only",
    )
    parser.add_argument("--llm-usd-per-mtok", type=float, default=None, help="prompt rate")
    parser.add_argument(
        "--llm-output-usd-per-mtok",
        type=float,
        default=None,
        help="generated-token rate; defaults to 4x the prompt rate if unset",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    from trigon.bpe import BPETokenizer
    from trigon.schema import SchemaCompiler
    from trigon.schema.tokens import CallableEstimator
    from trigon.types import SystemOneRequest
    from trigon.usecases import all_use_cases, use_case

    cases = [use_case(args.use_case)] if args.use_case else all_use_cases()
    tokenizer = BPETokenizer.load()
    # **The same tokenizer on both sides.** The compiler defaults to a
    # character heuristic, and the baseline is counted with the real BPE, so
    # leaving the default in place compares an estimate against an exact count
    # and calls the difference a saving. It inflated the prompted column by
    # roughly 40% here.
    #
    # This is the third time in this repository that two halves of one
    # pipeline using different tokenizers has been the defect --
    # `CLAUDE.md` documents it for backends, where it was a 500 on every
    # request. A cost model is the same shape of mistake with a quieter
    # failure: it produces a number, and the number is wrong.
    compiler = SchemaCompiler(estimator=CallableEstimator(tokenizer.encode, exact=True))

    print("Tokens per decision. Both columns are counted, not estimated.")
    print()
    print(
        f"{'use case':<16} {'typed':>7} {'typed':>8} {'prompted':>9} {'prompted':>9} "
        f"{'prompted':>9}"
    )
    print(f"{'':<16} {'cold':>7} {'cached':>8} {'cold':>9} {'cached':>9} {'output':>9}")
    print("-" * 64)

    rows = []
    for item in cases:
        compiled = compiler.compile_request(
            SystemOneRequest(state=item.state, questions=item.questions)
        )
        # Cold: the schema is paid on every request, which is what this
        # repository does today -- `Usage.cached_schema_tokens` reports 0 on
        # every response and says so in the spec. Cached: the schema prefix is
        # paid once and reused, which the layout permits and phase 3 builds.
        cold = compiled.total_tokens
        cached = compiled.state_tokens + compiled.readout_tokens
        prefix, per_request, reply = _prompt_for(item)
        prompt_cold = tokenizer.count(prefix) + tokenizer.count(per_request)
        prompt_cached = tokenizer.count(per_request)
        output_tokens = tokenizer.count(reply)
        rows.append((item, cold, cached, prompt_cold, prompt_cached, output_tokens))
        print(
            f"{item.name:<16} {cold:>7,} {cached:>8,} {prompt_cold:>9,} "
            f"{prompt_cached:>9,} {output_tokens:>9,}"
        )

    print()
    print("The typed path generates nothing: its output is logits, and a logit")
    print("is not billed. That column is the prompted path's alone.")
    print()
    print("Break-even $/MTok for the typed path, given the prompted path's rate R:")
    print()
    print(f"{'use case':<16} {'cold':>14} {'cached':>14}")
    print("-" * 46)
    for item, cold, cached, prompt_cold, prompt_cached, output_tokens in rows:
        # The prompted path pays prompt tokens at R and generated tokens at
        # `out_multiple` x R. Typed is cheaper while its own rate is below the
        # multiple printed. Like against like: cold against cold, cached
        # against cached.
        out_multiple = 4.0
        print(
            f"{item.name:<16} "
            f"{(prompt_cold + output_tokens * out_multiple) / cold:>12.2f}xR "
            f"{(prompt_cached + output_tokens * out_multiple) / cached:>12.2f}xR"
        )
    print()
    print("Read that as: the typed path wins while its rate is under that")
    print("multiple of the prompted path's prompt rate. Output tokens are")
    print("charged at 4x prompt here -- change it and the multiples move, which")
    print("is the point of printing a multiple rather than a price.")

    if args.typed_usd_per_mtok is not None and args.llm_usd_per_mtok is not None:
        out_rate = args.llm_output_usd_per_mtok
        if out_rate is None:
            out_rate = args.llm_usd_per_mtok * 4.0
        print()
        print("At YOUR rates -- these are your inputs, not this project's numbers.")
        print(
            f"  typed {args.typed_usd_per_mtok}/MTok, prompted {args.llm_usd_per_mtok}/MTok "
            f"in and {out_rate}/MTok out"
        )
        print()
        print(
            f"{'use case':<16} {'volume/mo':>12} {'typed $/mo':>12} "
            f"{'prompted $/mo':>14} {'saving':>9}"
        )
        print("-" * 68)
        for item, _cold, cached, _prompt_cold, prompt_cached, output_tokens in rows:
            n = item.monthly_volume
            typed = n * cached / 1e6 * args.typed_usd_per_mtok
            prompted = (
                n * prompt_cached / 1e6 * args.llm_usd_per_mtok + n * output_tokens / 1e6 * out_rate
            )
            print(
                f"{item.name:<16} {n:>12,} {typed:>12,.0f} {prompted:>14,.0f} "
                f"{prompted / typed if typed else float('inf'):>8.1f}x"
            )
        print()
        print("Typed uses the CACHED column, which this repository does not yet")
        print("do -- the layout permits it and phase 3 builds it. Quote the cold")
        print("column for what ships today.")

    if args.verbose:
        for item, *_ in rows:
            print()
            print(f"--- {item.name}: {item.purpose}")
            print(f"    corpus: {item.corpus}")
            if item.notes:
                print(f"    {item.notes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
