# The teacher-labelled synthetic-workflow stream

**Built, 2026-09-27.** This is the third of the five streams in `docs/data.md`:
LLM-generated (state, schema) pairs across twenty domains, labelled by a
teacher with its full distribution. `docs/data.md` says what it buys:
**coverage, not calibration.** This page records what was built, what it
cost, and what training on it measured. It also records what the teacher's
own probabilities look like when they can be checked against a computed
truth, which is the reason for the "not calibration".

**The headline:** the stream is built, and labelled cases cost **$0.62 per
1,000** on Modal. The teacher is measurably overconfident: on the verifiable
stream it is **75.0% accurate at 0.918 mean confidence, ECE 0.168 against a
noise-floor p95 of 0.011**. A Qwen2.5-1.5B student trained on the stream
learns the teacher's **position prior and very little else**: median argmax
agreement 0.530 across four seeds, against 0.527 for a predictor that ignores
the state. So the verdict is the modest one. The stream is built, and it is
cheap. It has not yet been shown to teach a student anything.

## What was built

| Piece | Where |
| --- | --- |
| Domains, the seeded plan, both prompt templates and their SHA-256, the parser, the loader's record-to-case step, and `teacher_agreement` | `src/trigon/evals/teacher.py` |
| `CorpusSpec` `teacher-workflows`: green, Apache-2.0, attribution, pinned build, and `teacher=` marking every label | `src/trigon/evals/corpora.py` |
| `Expectation.from_teacher`, and its refusal in `summarize()` (every ECE and gate) and in the calibrator fit | `src/trigon/evals/harness.py`, `src/trigon/cli.py` |
| Generation, labelling and the verifiable holdout on Modal (vLLM 0.11, one A10G per shard) | `scripts/modal_teacher.py` |
| The agreement-only training report | `scripts/train_corpus.py` (`teacher_main`) |
| The teacher's volume mounted into training jobs | `scripts/modal_train.py` |
| A 20-case sample of the published build, one case per domain (Apache-2.0 output) | `tests/fixtures/teacher_workflows_sample.jsonl` |
| Tests: loader, soft targets, the refusals, the licence tier, the fixture | `tests/test_teacher.py`, `tests/test_corpora.py` |

**The teacher** is `Qwen/Qwen2.5-7B-Instruct` at revision
`a09a35458c702b33eeacc393d103063234e8bc28`. It is Apache-2.0 in the card's
metadata and in the repository's `LICENSE` at that revision (checked
2026-09-27). The 3B and 72B instruct models are under the Qwen licence
instead, so the size is part of the licence statement. Apache-2.0 puts no
restriction on output, so the stream is **green**: it trains, evaluates and
may be redistributed. The data is still not committed. It lives on the
`trigon-teacher` Modal Volume at `tw0-n6000/cases.jsonl.gz`, pinned by SHA-256
`3a7032e19866d7b25ad647bf1df04bf3c721a3678317ffe8b782f8050beec5f7`.

**Two calls per case, both to the same teacher.**

1. **Generation.** `generation_plan(n, seed)` fixes, per case and from
   `(seed, index)` alone, the domain, a situation within it, the state's
   shape (prose, a JSON record, or a list of documents), whether the case is
   made deliberately borderline, whether options carry criteria, and one
   `(primitive, label count)` per question. The teacher is asked for exactly
   that, with a JSON skeleton built from the plan, and samples at
   temperature 0.8 with top-p 0.95 and a per-case seed. The output is
   validated by the contract itself (`DecisionRequest`); a case the gateway
   would refuse is not a training case.
2. **Labelling.** Each question is asked **alone**, never beside the case's
   other questions, over the state rendered exactly as the student sees it.
   For every declared label the teacher's log-likelihood of the whole reply
   is read off vLLM's `prompt_logprobs`: the label's tokens followed by
   `<|im_end|>`. The stored distribution is the softmax over those. Every
   record keeps the per-option log-probabilities, the token counts, the
   *declared mass* (how much of the teacher's reply probability landed on
   the declared strings at all), the raw generation, the plan, and the
   teacher, revision, engine and both template hashes. Nothing is an argmax.
   Scoring the full reply rather than its first token is what keeps "Yes"
   from being credited with the mass of "Yes, subject to some conditions".

Generation is deterministic by seed as far as vLLM allows. The plan is exact,
and the sampling seed is per case, but batched GPU arithmetic is not
bit-reproducible across batch compositions. A rebuild matches the plan
exactly and the text approximately; the SHA-256 pins the build that was
measured.

## The domains

Twenty, rotated so each gets a twentieth of the build (253–289 kept each).
The first three are the use cases `trigon.usecases` prices and
`trigon.evals.workflows` runs. The other seventeen are what the green corpora
do not cover.

| | | | |
| --- | --- | --- | --- |
| support_triage | content_moderation | product_reviews | ecommerce_returns |
| insurance_claims | recruiting | contract_review | payments_risk |
| it_incidents | code_review | clinical_intake (administrative, not diagnosis) | real_estate |
| travel_bookings | education | sales_leads | logistics |
| access_requests | ad_compliance | public_services | accounts_payable |

## The build: `tw0-n6000`

| | |
| --- | ---: |
| Cases planned | 6,000 |
| Cases kept (a valid request of the planned primitives) | **5,558** (92.6%) |
| Questions labelled | 18,740 |
| — Choice / Noul / Score | 7,448 / 6,426 / 4,866 |
| Replies scored (one per declared label) | 71,736 |
| Tokens generated | 2,906,941 |
| Questions per case, 1 → 6 | 589 / 1,096 / 1,384 / 1,160 / 807 / 522 |
| Choice options | 2 to 16 (2–12 planned) |
| Score levels | 2 to 10 (3–7 planned) |
| Cases whose schema came back exactly as planned | 73.7% |
| Choice and Score questions with the planned label count | 84.7% |
| State shape: prose / record / documents | 2,421 / 1,980 / 1,157 |
| Planned borderline | 39.8% |
| Split, by a hash of the case id | 3,860 train / 1,698 held out (5,714 questions) |

Rejected, of 6,000: 163 returned primitives other than the plan's, 103 were
not valid JSON, 92 failed the contract (a duplicate option name, too few
options), 83 had a question id that was not snake_case, and 1 returned no
JSON. The first smoke test rejected 15 of 24: the prompt showed one fixed
example shape (choice, noul, score) and the teacher copied its order instead
of the plan's. The skeleton is now built from the plan, which took the
rejection rate from 62% to 7%.

**What the teacher's labels look like.** Mean top-label probability 0.938;
**64.6% of questions above 0.99** (Noul 76.5%, Choice 68.3%, Score 43.3%),
and 13.0% below 0.8. The declared mass has a median of 0.9995, but 6.8% of
questions put under half of the reply mass on the declared strings. That
happens mostly on Nouls, where the teacher prefers "Yes" to "yes". There the
stored distribution is the teacher's choice *between the declared labels*,
which is what the student is asked, and the declared mass records how much
the teacher would rather have said something else.

## The teacher against a computed truth — calibration, measured once

The one place a teacher's probabilities can be scored as calibration is
where the labels are not its own. `label_verifiable` had the same teacher, with
the same label template, answer 1,700 cases of `synthetic_outcome_cases`
(noise 0, so the truth is the predicate) and the first step of the two
committed workflows (600 cases each, outcomes computed by their generators).

| Verifiable set | n | Teacher accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE | Floor p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `synthetic`, all three questions | 5,100 | 0.7500 | 0.9176 | **+0.1676** | **0.1676** | 0.1676 | 0.0109 |
| — `plan` (Choice, stated in the record) | 1,700 | 1.0000 | 1.0000 | 0.0000 | — | — | — |
| — `at_risk` (Noul, rule in the question) | 1,700 | 0.9571 | 0.9908 | +0.0337 | — | — | — |
| — `size` (Score, threshold not stated) | 1,700 | 0.2929 | 0.7621 | **+0.4691** | — | — | — |
| `support_triage` routing | 600 | 1.0000 | 0.9953 | −0.0047 | — | — | — |
| `moderation_queue` gate | 600 | 0.8550 | 0.9992 | **+0.1442** | — | — | — |

The pooled synthetic ECE is quotable: n = 5,100 clears
`MIN_CALIBRATION_SAMPLES`, and 0.168 is fifteen times the floor, so the
miscalibration is real rather than sampling noise. The per-question rows and
the two workflow rows are below the sample floor, so their ECE is not quoted.
Their accuracy and mean confidence are. The teacher is right where the answer
is written down, and confidently wrong where it has to infer. It says 0.76 on
a seat-count tier it gets right 29% of the time, and 0.999 on a moderation
gate it gets right 85.5% of the time. This is the measured version of the
sentence in `docs/data.md`, and it is why the harness refuses these labels
wherever an ECE would be published.

## A student trained on it: Qwen2.5-1.5B, four seeds

`python scripts/modal_train.py launch --corpus teacher-workflows -n 0 --epochs 3
--extra "--backbone qwen2.5-1.5b --lr 1e-4"`: LoRA rank 16, three epochs over
all 3,860 training cases, soft cross-entropy against the teacher's
distribution, run two seeds at a time. The report is agreement on the 1,698
held-out cases (5,714 questions). **No calibrator was fitted and no gate was
read**, because both would be fitted to or scored against the teacher.

| Seed | Argmax agreement | − ignores input | KL(teacher ‖ student) | − ignores input | Choice / Noul / Score agreement | Mean confidence | ECE vs a teacher draw | Adaptive | Floor p95 |
| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 0 | 0.5380 | +0.0110 | 0.8990 | −0.0214 | 0.481 / 0.667 / 0.458 | 0.5278 | 0.0260 | 0.0269 | 0.0214 |
| 1 | 0.5289 | +0.0019 | 0.9116 | −0.0088 | 0.478 / 0.666 / 0.429 | 0.5395 | 0.0155 | 0.0204 | 0.0211 |
| 2 | 0.5296 | +0.0026 | 0.8997 | −0.0208 | 0.482 / 0.663 / 0.429 | 0.5225 | 0.0145 | 0.0190 | 0.0212 |
| 3 | 0.5299 | +0.0030 | 0.9109 | −0.0096 | 0.477 / 0.666 / 0.434 | 0.5101 | 0.0190 | 0.0195 | 0.0209 |
| **median** | **0.5298** | **+0.0028** | **0.9053** | **−0.0152** | | 0.5252 | 0.0173 | | |
| ignores its input | 0.5270 | | 0.9204 | | 0.479 / 0.666 / 0.419 | | | | |
| lexical floor | 0.4697 | | 1.0505 | | | | | | |
| the teacher itself | 1 | | 0 | | | 0.9362 | | | |

Per-seed reports: `qwen15b-e3-seed{0,1,2,3}.md` and `.json` beside this
file. The training curves are in `-training.json`, and the four containers
are in `qwen15b-e3-modal-run.json`. Every seed exits 0, the script's one test:
it beats the input-ignoring predictor on both agreement and KL.

"Ignores its input" is the predictor that answers, per (primitive, label
count), the position the teacher chose most often on the training split, and
reports the mean teacher distribution for KL. It is strong here. The teacher
picks early options disproportionately, so position alone agrees with it
52% of the time.

**What the four seeds say, on their median and spread:**

- **The agreement lift is not distinguishable from zero.** +0.28 points at
  the median, +0.19 to +1.10 across seeds, on 5,714 questions, where one
  standard error of an agreement rate is about 0.66 points. The spread is
  narrow; the effect is small.
- **KL moves the right way on every seed**, by 0.009–0.021 nats (1–2%). That
  is consistent, but it is small.
- **All of it is Score.** Choice agreement is the baseline's to the third
  decimal on every seed (0.477–0.482 against 0.479), and Noul's is the
  baseline's (0.663–0.667 against 0.666). Score gains 1–4 points. A Score
  question reads out through one slot for all its levels
  (`SCORE_READOUT_PER_LEVEL` is off), so on Score the student can learn
  "the teacher prefers the middle level" as a function of the level count,
  and that is the gain. That is believed, not measured.
- **The student is far less sure than the teacher**: mean confidence 0.51–0.54
  against 0.94. Its ECE against a label drawn from the teacher (0.0145–0.0260,
  floor p95 0.021) is at or just above the floor, and it measures one thing:
  a student that reports roughly the teacher's prior matches the prior's hit
  rate. It is *not* calibration, and a student that does learn to imitate
  will move toward the teacher's 0.94 confidence, whose own ECE against truth
  is 0.168.

**Why it did not learn more is not measured.** The candidates are recorded
here so the next run tests one rather than proposing it as a finding:
3,860 cases in which every case carries a new schema, where Banking77 had
9,000 cases of one schema; three epochs at 1e-4 with LoRA rank 16, the
Banking77 recipe (though not for want of epochs: validation loss bottomed
at epoch 2 on every seed and rose in epoch 3 on three of four, and the
trainer keeps the best epoch); a teacher whose labels on generated cases may be
noisier than its confidence suggests (see *not claimed*); and the position
bias itself, which is a cheap optimum for soft cross-entropy on a sharp
teacher.

## What it cost

| | Container time | Dollars |
| --- | ---: | ---: |
| Generation and labelling, 4 shards on A10G | 8,078 s (2.24 h): load 326 s, generate 3,354 s, label 4,396 s | **$3.47** |
| **Per 1,000 labelled cases kept** | | **$0.62** |
| Per 1,000 labelled questions | | $0.19 |
| Verifiable holdout, 6,300 questions | 780 s | $0.34 |
| Smoke tests, four aborted, two complete | | ≈ $0.5 |
| Student, four seeds on A10G | 5,737 s (1.59 h), 1,426–1,443 s a seed | $1.76 (GPU list rate only) |
| **Total Modal spend for this build** | | **≈ $6.1** |

Priced at Modal's list rates on 2026-09-27: A10G $0.000306/s, plus 4 CPU
cores and 32 GiB, together $0.000429/s per shard (`PRICE_PER_SECOND` in the
script). Container time is measured inside the function, from before the
model loads to after the shard is written, so it omits the few seconds of
container start. Labelling cost more than generation. With the plan's mix,
each case asks about 3.4 questions of about 3.8 labels each, and each label
is a separate full-reply scoring pass over the prompt.

## Commands

```bash
python scripts/modal_teacher.py launch --n 6000 --shards 4 --only 0,1500   # 2 GPUs
python scripts/modal_teacher.py launch --n 6000 --shards 4 --only 3000,4500
python scripts/modal_teacher.py launch --n 0 --build tw0-n6000 --verifiable 1700
python scripts/modal_teacher.py merge tw0-n6000      # prints the SHA-256 to pin
python scripts/modal_teacher.py cost tw0-n6000
python scripts/modal_teacher.py fetch tw0-n6000      # into corpora/, ignored by git
python scripts/modal_teacher.py fixture tw0-n6000
python scripts/modal_train.py launch --corpus teacher-workflows --seeds 0,1 -n 0 \
    --epochs 3 --prefix qwen15b-e3 --extra "--backbone qwen2.5-1.5b --lr 1e-4"
python scripts/modal_train.py collect <run id> --out-dir reports/teacher
```

The build was launched at `a1152fd` (shards 0 and 1500), `9a17c03`
(shards 3000 and 4500) and `70b3ca5` (the verifiable holdout). The workspace's GPU cap was lowered mid-build, so
two shards were cancelled while still queued and relaunched after the first
two finished. The generation and labelling code is identical at both
commits, and every shard's summary on the volume records its own commit.

## What is not claimed

- **Nothing here is calibration evidence for the student.** The student
  table measures imitation. A student that matched the teacher exactly would
  inherit an ECE of 0.168 on the one set where the teacher's calibration
  could be checked.
- **The teacher's labels are not checked for correctness at scale.** The
  20-case fixture was read by the agent that built it, not by a person.
  About 3 of its 64 answers look wrong, each at high confidence (0.78 to
  0.98). For example, a review that says video calls are poor gets "sound
  satisfactory for music and video calls: yes" at 0.98, and a listing $300
  over its market rate gets "not overpriced" at 0.78. About five of its Score
  questions have levels that are not in order, such as
  `mild, moderate, low, high, …` or a risk scale written high to low. The
  contract cannot catch that, and the generator should be made to.
- **Transfer to Banking77 is measured, and negative** (`transfer.md`, added
  2026-09-28). Used as the init for the certified Banking77 recipe, these
  students left the ln 77 plateau later than the base model. At 1,000 cases
  the median accuracy was 0.0895 against 0.6549. At the full budget one seed
  of four was still at chance (median 0.8787, against 0.9009 certified). The
  synthetic suite is still unrun.
- **The domains are one teacher's idea of them.** The states read as
  plausible and generic: many are about a "John Doe". Diversity was set by
  the plan, not measured against real traffic.
- **Determinism is by plan and seed, not bit-for-bit.** See above.
