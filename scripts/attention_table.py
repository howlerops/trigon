#!/usr/bin/env python
"""Generate the attention-cost table in docs/architecture.md.

The table is the evidence for the claim the budgets in ``trigon.limits`` are
built on -- that schema is nearly free to grow and state is the one axis that
costs quadratically. A hand-written table is an assertion; this makes it a
measurement, and ``tests/test_attention_table.py`` fails if the checked-in
table and this script disagree.

Run it after any change to the layout, the mask, or the token estimator:

    python scripts/attention_table.py            # print the table
    python scripts/attention_table.py --write    # update docs/architecture.md
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from trigon.limits import DEFAULT_BUDGET  # noqa: E402
from trigon.schema.compiler import SchemaCompiler  # noqa: E402
from trigon.types import ChoiceQuestion, OptionSpec, SystemOneRequest  # noqa: E402

DOC = pathlib.Path(__file__).resolve().parent.parent / "docs" / "architecture.md"
START = "<!-- attention-table:start -->"
END = "<!-- attention-table:end -->"

#: How the schema budget might be split across questions. The cost is not the
#: same for each: the per-question term is a square, so few large questions
#: cost far more than many small ones at the identical token count.
SHAPES = (6, 16, 64, 256, 1024)

#: (state tokens, questions, options per question). Chosen to bracket the
#: regime change: state-dominated, schema-dominated, and many questions.
ROWS = ((32_768, 4, 20), (8_192, 20, 50), (8_192, 64, 50))

_CRITERIA = "a described situation the option covers"


def _state_of(target_tokens: int, compiler: SchemaCompiler) -> str:
    """Filler text the compiler counts as (almost exactly) ``target_tokens``."""
    word = "situation "
    low, high = 1, target_tokens * 4
    while low < high:
        mid = (low + high) // 2
        if compiler.estimator.count(word * mid) < target_tokens:
            low = mid + 1
        else:
            high = mid
    return word * low


def measure(state_tokens: int, n_questions: int, n_options: int) -> dict:
    compiler = SchemaCompiler()
    request = SystemOneRequest(
        state=_state_of(state_tokens, compiler),
        questions={
            f"q{i}": ChoiceQuestion(
                instructions="Pick the option that fits this case.",
                options=[
                    OptionSpec(name=f"option_{j:04d}", criteria=_CRITERIA) for j in range(n_options)
                ],
            )
            for i in range(n_questions)
        },
    )
    compiled = compiler.compile_request(request)
    return {
        "label": (
            f"{compiled.state_tokens // 1024}k state, {n_questions} questions x {n_options} options"
        ),
        "tokens": compiled.total_tokens,
        "saving": compiled.attention_saving,
        "dense": compiled.dense_equivalent_tokens,
    }


def envelope() -> list[dict]:
    """What a request at the published ceilings costs, by question count.

    Computed from ``trigon.limits`` rather than compiled, because a real
    request of this size takes minutes to build and the arithmetic is the
    claim. One readout slot per question -- dot-product scoring, which is what
    makes option sets this large affordable in the first place.
    """
    state = DEFAULT_BUDGET.state_tokens
    rows = []
    for n_questions in SHAPES:
        per_question = min(
            DEFAULT_BUDGET.schema_tokens // n_questions, DEFAULT_BUDGET.max_question_tokens
        )
        pairs = n_questions * (per_question**2 + (per_question + state + 1)) + state**2
        rows.append(
            {
                "questions": n_questions,
                "per_question": per_question,
                "tokens": per_question * n_questions + state + n_questions,
                "dense": math.isqrt(pairs),
            }
        )
    return rows


def render() -> str:
    rows = [measure(*row) for row in ROWS]
    out = [
        "| Layout | Tokens | Saving vs dense | Costs like dense |",
        "| --- | ---: | ---: | ---: |",
    ]
    out += [
        f"| {r['label']} | {r['tokens']:,} | {r['saving']:.1%} | {r['dense']:,} |" for r in rows
    ]
    out += [
        "",
        "At the published ceilings, what the request costs depends on how the",
        "schema budget is split -- the per-question term is a square, so a few",
        "large questions cost far more than many small ones at the same token",
        "count:",
        "",
        "| Questions | Tokens each | Total tokens | Costs like dense |",
        "| ---: | ---: | ---: | ---: |",
    ]
    out += [
        f"| {r['questions']:,} | {r['per_question']:,} | {r['tokens']:,} | {r['dense']:,} |"
        for r in envelope()
    ]
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="update docs/architecture.md in place")
    args = parser.parse_args()

    table = render()
    if not args.write:
        print(table)
        return 0

    text = DOC.read_text()
    before, _, rest = text.partition(START)
    _, _, after = rest.partition(END)
    DOC.write_text(f"{before}{START}\n{table}\n{END}{after}")
    print(f"wrote {DOC}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
