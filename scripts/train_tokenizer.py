#!/usr/bin/env python
"""Train the byte-level BPE vocabulary that ships in the package.

Run once, and again only when the corpus or the vocabulary size changes:

    python scripts/train_tokenizer.py                    # write src/trigon/data/bpe.json
    python scripts/train_tokenizer.py --vocab-size 16384

Needs the ``tokenizers`` library, which is a build-time dependency and
deliberately not a runtime one -- ``trigon.bpe`` encodes in pure Python so the
gateway, the schema compiler and CI can count tokens without it. This script is
the only thing that imports it.

**What it trains on, and why that is a defensible corpus.** Everything the
reference model is evaluated against plus the repository's own prose: the
synthetic outcome corpus, the jaggedness benchmarks, the workflow suites, the
cardinality probe's generated option names, and the Markdown in `docs/`. All of
it is text this project generates or wrote, so the vocabulary carries no
licence question of its own -- which matters, because a tokenizer trained on a
scraped corpus inherits that corpus's terms and `docs/data.md` takes that
seriously everywhere else.

It also means the vocabulary is **fitted to this corpus and no other**, and
that is a real limitation rather than a footnote: it is a good encoding for the
text the eval harness produces and an ordinary one for anything else. Phase 1
replaces it with the backbone's own tokenizer through the same
``CallableEstimator`` seam. What it buys in the meantime is an injective
vocabulary -- the hashing tokenizer collides 24.4% of this same corpus.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from trigon.bpe import RESERVED  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent.parent / "src" / "trigon" / "data" / "bpe.json"


def corpus() -> list[str]:
    """Every string the model is trained or evaluated on. Not the docs -- see below."""
    from trigon.evals.cardinality import build_cardinality_probe
    from trigon.evals.datasets import synthetic_outcome_cases
    from trigon.evals.jaggedness import all_benchmarks
    from trigon.evals.workflows import all_workflows

    lines: list[str] = []

    def absorb(request) -> None:
        lines.append(str(request.state))
        for question in request.questions.values():
            lines.append(question.instructions)
            for member in getattr(question, "options", None) or getattr(question, "levels", []):
                lines.append(member.name)
                if member.criteria:
                    lines.append(member.criteria)

    for case in synthetic_outcome_cases(n=4000, seed=0, noise=0.2):
        absorb(case.request)
    for benchmark in all_benchmarks(n=200, seed=0):
        for case in benchmark.cases():
            absorb(case.request)
    for _, cases in all_workflows(n=200, seed=0):
        for case in cases:
            lines.append(str(case.state))
    # The use-case schemas: their states are prose in the domains the model is
    # meant to serve, and their label sets are what a caller actually sends.
    # This is data, not writing about data.
    from trigon.usecases import all_use_cases

    for item in all_use_cases():
        lines.append(str(item.state))
        lines.append(item.purpose)
        for question in item.questions.values():
            lines.append(question.instructions)
            for member in getattr(question, "options", None) or getattr(question, "levels", []):
                lines.append(member.name)
                if getattr(member, "criteria", None):
                    lines.append(member.criteria)

    question, queries, _ = build_cardinality_probe(n_options=4096, n_queries=200, seed=0)
    lines.extend(option.name for option in question.options)
    lines.extend(option.criteria for option in question.options if option.criteria)
    lines.extend(queries)

    # **The prose is deliberately absent, and it used to be here.**
    #
    # The model never sees documentation. It sees state, instructions and
    # label names, all of which this function already collects. Training the
    # vocabulary on docs spent budget on text the model will never encounter
    # -- and, far worse, coupled a committed generated artifact to prose.
    #
    # Every checkpoint records the vocabulary it was trained against and
    # refuses to load under a different one, so editing a markdown file
    # invalidated every checkpoint in the repository. That is not theoretical:
    # this vocabulary went 5,635 -> 4,712 -> 6,392 -> 4,776 across one working
    # session, and the single most important measurement in the `size`
    # investigation -- the one run where that question beat its marginal --
    # turned out to have been made against a vocabulary that a later
    # documentation commit destroyed. It did not reproduce, and the reason it
    # did not reproduce was this function.
    #
    # A generated artifact may depend on the data. It may not depend on the
    # writing about the data.
    return [line for line in lines if line]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    lines = corpus()
    print(
        f"training on {len(lines):,} strings, {sum(map(len, lines)):,} characters",
        file=sys.stderr,
    )

    tokenizer = Tokenizer(models.BPE())
    # add_prefix_space=False keeps "pro" and " pro" distinct, which matters:
    # an option name is emitted without a leading space and the same word
    # inside the state has one.
    # Digits individually, then byte level -- the same pre-tokenization
    # `trigon.bpe._PIECE` applies at encode time. BPE merges within a piece and
    # never across one, so training with `\d+` pieces and encoding with `\d`
    # pieces builds a vocabulary full of whole-number tokens the encoder can
    # never emit, and leaves the numbers it *can* emit undertrained.
    #
    # That mismatch is what hid the numbers from the model in the first place:
    # `127` and `128` trained as single tokens 2030 and 2262, two unrelated
    # embedding rows, and the threshold question over `seats` sat at chance on
    # every seed. `tests/test_bpe.py` asserts this file and the encoder agree.
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.ByteLevel(add_prefix_space=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        lines,
        trainers.BpeTrainer(
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            # Every single byte, so no input is unencodable and the pure-Python
            # encoder never has to invent an unknown token.
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=list(RESERVED),
            show_progress=False,
        ),
    )

    payload = json.loads(tokenizer.to_str())
    vocab: dict[str, int] = payload["model"]["vocab"]
    merges = payload["model"]["merges"]
    merges = [" ".join(m) if isinstance(m, list) else m for m in merges]

    for expected_id, token in enumerate(RESERVED):
        if vocab.get(token) != expected_id:
            raise SystemExit(f"reserved token {token!r} landed at id {vocab.get(token)}")

    target = pathlib.Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "vocab_size": len(vocab),
                "reserved": list(RESERVED),
                "vocab": vocab,
                "merges": merges,
            },
            ensure_ascii=False,
            indent=0,
            sort_keys=False,
        )
    )
    print(f"wrote {target} — {len(vocab):,} tokens, {len(merges):,} merges", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
