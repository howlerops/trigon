# Pricing a use case

`python scripts/price.py` quotes what one decision costs, per use case, with
the measured part and the assumed part kept apart. Run it against your own
rates; the defaults quote no money at all.

## The finding, stated first because it corrects this project's own story

**Per request, the typed path sends *more* tokens than a prompted LLM, not
fewer.** Counted like for like — the same tokenizer on both sides, both caching
what they can cache:

| Use case | typed, cached | prompted, cached | prompted output |
| --- | ---: | ---: | ---: |
| moderation | 80 | 74 | 40 |
| sentiment | 202 | 196 | 40 |
| support_triage | 136 | 130 | 38 |

The relationship is exact, not approximate: **80 = 74 + 6**, **202 = 196 + 6**,
**136 = 130 + 6**. Both paths read the same state; the typed path then pays one
readout slot per question, and there are two questions plus a Noul in each of
these, so six slots. That is the whole of its token disadvantage and it is a
constant, not a rate.

What it buys for those six tokens is that **it generates nothing**. Its output
is logits, and a logit is not billed. At a typical 4× markup on generated
tokens, the 38–40 tokens of JSON the prompted path must produce are worth
~160 token-equivalents — an order of magnitude more than the six it costs to
avoid them.

Break-even, which needs no price at all to state:

| Use case | cold | cached |
| --- | ---: | ---: |
| moderation | 1.96×R | 2.92×R |
| sentiment | 1.60×R | 1.76×R |
| support_triage | 1.76×R | 2.07×R |

Read: the typed path is cheaper while its own $/MTok is below that multiple of
the prompted path's prompt rate. **So the architecture justifies roughly a 2×
price premium, not a 100× one.**

Everything beyond 2× has to come from **$/token**, which is a function of model
size and is the one number this project has never measured. `docs/roadmap.md`
records the inherited figure — $0.80/hr on an L4 at ~30k prefill tok/s, so
~$0.007/MTok — as arithmetic over unsourced inputs, pending a burn-in. That
burn-in is not a nice-to-have for the cost story. **It is the cost story.** The
token layout contributes a factor of two; the remaining two orders of magnitude
are entirely the claim that a small typed model serves tokens far more cheaply
than a frontier one, and that claim is unverified.

At $0.007 against $0.25/MTok the tool prints savings of 75–96×. Those are
*your* inputs multiplied by *our* exact token counts. The token counts are
trustworthy. The 0.007 is not, yet.

**Two corrections this table has already survived.** The first draft compared a
*cached* typed path against an *uncached* prompted one, because the prompted
path's instruction block is equally cacheable on any provider with prompt
caching and that was overlooked; fixing it cut the headline by a third. The
second counted the typed path with the compiler's character heuristic and the
prompted path with the real BPE, comparing an estimate against an exact count —
which inflated the prompted column by about 40%. Both errors ran in our favour,
which is the direction errors run when nobody is looking for them.

## What is measured and what is not

| Quantity | Status |
| --- | --- |
| Tokens per request, typed path | **Measured.** `SchemaCompiler`, exact |
| Tokens per request, prompted path | **Measured.** A real prompt through the same tokenizer |
| Generated tokens, prompted path | **Measured.** The JSON reply, counted |
| Attention pairs per request | **Measured.** `scripts/attention_table.py` |
| Gateway overhead | **Measured.** 2.28 ms, 1.5% of the p50 budget |
| $/MTok, typed path | **Assumed.** Needs the L4 burn-in |
| $/MTok, prompted path | **Yours.** A published list price, not ours to quote |
| Throughput per GPU | **Unmeasured.** Needs hardware |

## How the comparison is kept fair

**The baseline is the strongest form of the alternative.** One call, structured
output, all questions at once — not N separate calls, and not
chain-of-thought. A baseline chosen to lose is not a baseline.

**Both sides cache.** The typed path's schema prefix is cacheable, which is the
architectural claim `tests/test_independence.py` asserts. The prompted path's
instruction block is equally cacheable on any provider with prompt caching. The
first draft of `scripts/price.py` compared a cached typed path against an
uncached prompted one, which inflated the ratio from ~2.1× to ~3.4×. Comparing
like against like cost the headline a third of its size and is the only version
worth quoting.

**The cached typed column is not what ships today.** This repository holds no
KV cache: `Usage.cached_schema_tokens` reports 0 on every response and says so
in the spec. Quote the cold column for what exists, the cached column for what
phase 3 builds.

## Use cases

`src/trigon/usecases.py` carries the schemas, each naming the **green-tier**
corpus from `docs/data.md` it would be trained and evaluated on — so a use case
is a claim about something buildable rather than a demo:

- **sentiment** — polarity (Choice), stars (Score), needs-a-reply (Noul) over
  one read of a review. GoEmotions (Apache-2.0), HelpSteer2 (CC BY 4.0).
- **moderation** — action (Choice), severity (Score), targets-a-person (Noul).
  Civil Comments (CC0-1.0), measuring_hate_speech (CC BY 4.0).
- **support_triage** — team (Choice), urgency (Score), wants-refund (Noul).
  Banking77 (CC BY 4.0), CLINC150 (CC BY 3.0).

Classic SST-5 is deliberately absent: its card carries no licence, which the
audit reads as red.

The states are illustrative and written by hand to be the right shape and
length for pricing. The labelled corpora are a phase-2 deliverable and are not
vendored here, so **no accuracy claim attaches to these use cases yet** — they
price the contract, they do not evaluate the model.
