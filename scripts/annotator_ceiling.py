#!/usr/bin/env python
"""How much of HelpSteer2 is predictable at all? Oracles built from the annotators.

    python scripts/annotator_ceiling.py --out reports/helpsteer2/ceiling.md

Both models fail HelpSteer2's `accuracy_over_baseline` gate (+0.05): the
spike at +0.0188, Qwen2.5-1.5B at +0.0259. Before a third model is trained
against that gate, this asks what the best possible predictor could score --
using not the text but the *other annotators' ratings of the same response*,
which is more information about a rating than any model of the text has.

Three oracles, on the `disagreements/` split's held-out quarter:

1. **The naive one** -- predict the mode of every annotator's rating,
   including the annotator whose rating is the label. It scores well and is
   wrong: the label votes for itself. It is printed because it was computed
   first and nearly believed.
2. **Leave-one-out with a prior** -- predict from the *other* annotators'
   votes plus the population's marginal, the best fair use of the panel for
   predicting one annotator. This is the ceiling for `helpsteer2-annotators`,
   whose label is one annotator drawn per case.
3. **Split half** -- on items with four or more annotators, how well does one
   half-panel's rounded mean predict the other half's? A proxy for how
   reproducible an *aggregated* label is, which is what the original
   `helpsteer2` corpus scores against. Half-panels are noisier than the full
   panels behind that corpus's labels, so this bounds the question loosely,
   not exactly.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import pathlib
import random
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    from trigon.evals.corpora import _held_out, corpus, fetch

    spec = corpus("helpsteer2-annotators")
    fields = spec.score_fields
    rows = [json.loads(line) for line in gzip.open(fetch(spec)["all"], "rt")]
    usable = [
        (i, r)
        for i, r in enumerate(rows)
        if all(isinstance(r.get(q), list) and r[q] for q in fields)
    ]

    def drawn(i: int, q: str, ratings: list[int]) -> int:
        # The same annotator the corpus loader draws for this case.
        digest = hashlib.blake2b(f"{spec.name}/{i}/{q}".encode(), digest_size=4).digest()
        return int.from_bytes(digest, "big") % len(ratings)

    train = [(i, r) for i, r in usable if not _held_out(spec, r)]
    test = [(i, r) for i, r in usable if _held_out(spec, r)]

    table = []
    for q in fields:
        prior = collections.Counter(r[q][drawn(i, q, r[q])] for i, r in train)
        total = sum(prior.values())
        modal = prior.most_common(1)[0][0]
        naive, loo, marginal = [], [], []
        for i, r in test:
            ratings = r[q]
            d = drawn(i, q, ratings)
            label = ratings[d]
            everyone = collections.Counter(ratings)
            naive.append(max(range(spec.levels), key=lambda k: (everyone[k], -k)) == label)
            others = collections.Counter(ratings[:d] + ratings[d + 1 :])
            score = {k: others[k] + 2.0 * prior[k] / total for k in range(spec.levels)}
            loo.append(max(score, key=score.get) == label)
            marginal.append(label == modal)
        table.append((q, statistics.mean(marginal), statistics.mean(naive), statistics.mean(loo)))

    rng = random.Random(0)
    big = [r for _, r in usable if all(len(r[q]) >= 4 for q in fields)]

    def rounded(xs):
        return int(sum(xs) / len(xs) + 0.5)

    halves = []
    for q in fields:
        pairs = []
        for r in big:
            xs = list(r[q])
            rng.shuffle(xs)
            h = len(xs) // 2
            pairs.append((rounded(xs[:h]), rounded(xs[h : 2 * h])))
        modal = collections.Counter(b for _, b in pairs).most_common(1)[0][0]
        halves.append(
            (
                q,
                statistics.mean(b == modal for _, b in pairs),
                statistics.mean(a == b for a, b in pairs),
            )
        )

    def pooled(values):
        return sum(values) / len(values)

    lines = [
        "# HelpSteer2: how predictable the ratings are, measured from the annotators",
        "",
        f"**{spec.attribution}**",
        "",
        "`scripts/annotator_ceiling.py`. The `disagreements/` split: "
        f"{len(usable):,} pairs with every annotator's rating, {len(test):,} held out "
        "by prompt hash exactly as `helpsteer2-annotators` holds them out.",
        "",
        "## Predicting one annotator's rating (the `helpsteer2-annotators` target)",
        "",
        "| Question | Marginal | Naive oracle | Lift | Leave-one-out oracle | **Lift** |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for q, m, n, lo in table:
        lines.append(f"| `{q}` | {m:.4f} | {n:.4f} | {n - m:+.4f} | {lo:.4f} | **{lo - m:+.4f}** |")
    pm, pn, pl = (pooled([row[k] for row in table]) for k in (1, 2, 3))
    lines += [
        f"| pooled | {pm:.4f} | {pn:.4f} | {pn - pm:+.4f} | {pl:.4f} | **{pl - pm:+.4f}** |",
        "",
        "The naive oracle counts the labelling annotator's own vote, and most of its",
        "lift is that. The leave-one-out oracle predicts from the other annotators",
        "on the same response plus the population prior -- more information about",
        "the rating than a model of the text has -- and its pooled lift is the",
        "ceiling for this target.",
        "",
        "## Reproducing an aggregated label (the `helpsteer2` target, loosely)",
        "",
        f"{len(big):,} pairs with four or more annotators on every question. One",
        "half-panel's rounded mean predicting the other half's.",
        "",
        "| Question | Marginal | Half predicts half | **Lift** |",
        "| --- | ---: | ---: | ---: |",
    ]
    for q, m, a in halves:
        lines.append(f"| `{q}` | {m:.4f} | {a:.4f} | **{a - m:+.4f}** |")
    hm, ha = pooled([h[1] for h in halves]), pooled([h[2] for h in halves])
    lines += [
        f"| pooled | {hm:.4f} | {ha:.4f} | **{ha - hm:+.4f}** |",
        "",
        "Half-panels of two or three are noisier than the full panels behind the",
        "aggregated labels, so this is a loose bound: a full-panel label is more",
        "reproducible than this, and a model could in principle beat it.",
        "",
    ]
    report = "\n".join(lines)
    print(report)
    if args.out:
        pathlib.Path(args.out).write_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
