# Banking77 — the first certified result on data the generator did not write

Banking77 (Casanueva et al., 2020), PolyAI. CC BY 4.0.
https://github.com/PolyAI-LDN/task-specific-datasets

Four seeds, 7,083 training cases, 1,000 held out to fit the calibrator, 6
epochs, d_model 128, 2 layers. Evaluated on 5,000 cases: the corpus's own
3,080-row test split plus 1,920 rows held out of train, because the test split
alone is below `MIN_CALIBRATION_SAMPLES` and an ECE below that floor is not
evidence.

**Every seed clears every blocking gate.**

| Seed | Accuracy | Marginal | Lift | ECE | Adaptive ECE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.7320 | 0.0160 | **+0.7160** | 0.0269 | 0.0257 |
| 1 | 0.7176 | 0.0166 | **+0.7010** | 0.0361 | 0.0382 |
| 2 | 0.7126 | 0.0174 | **+0.6952** | 0.0185 | 0.0183 |
| 3 | 0.7404 | 0.0162 | **+0.7242** | 0.0177 | 0.0212 |

Median accuracy 0.7248, range 0.7126–0.7404. Median ECE 0.0227, range
0.0177–0.0361.

Accuracy is the calibrated run's, the same one the gates read. That matters on
seed 2 and nowhere else — see below.

## Why this one counts and the synthetic runs do not

**The baseline is real.** A 77-way choice where the marginal predictor scores
1.6%. On the synthetic generator's four-option questions a model that ignores
its input scores 25%, and `accuracy_over_baseline` has to work hard to reject
it. Here it cannot be passed by accident.

**The number is above its floor.** A perfectly calibrated model scores ECE
0.0153 on this run (95th percentile 0.0208) and the measured error is
0.0177–0.0361, so `gate_is_testable` passes and these ECEs mean something. The
two-seed pilot in `pilot/` did not clear that bar — it evaluated on 1,500 rows
where a perfect model scores 0.0221 against a 0.05 gate — and its calibration
numbers are not quoted anywhere for that reason.

**The spread is narrow.** This is the part worth pausing on. Seed variance has
been this project's recurring finding: the synthetic configuration decides its
own outcome by seed, `size` never escaped its marginal on any of seven
interventions, and `at_risk` collapsed to exactly its marginal on one draw of
four. Here four seeds land within 2.3 points of each other and all four
certify. `CLAUDE.md` says to prefer a narrow spread to a good maximum; this is
the first configuration in the repository that has one.

## The calibrator declined itself on three seeds of four

| Seed | Choice head |
| ---: | --- |
| 0 | **none** — unscaled 0.0622, best fit 0.0582 |
| 1 | **none** — unscaled 0.0527, best fit 0.0414 |
| 2 | isotonic, 0.0642 → 0.0229 |
| 3 | **none** — unscaled 0.0674, best fit 0.0519 |

Three seeds served **unscaled** and still passed the ECE gate. The
paired-bootstrap rule requires a calibrator to be *demonstrably* better at 95%
confidence before it is applied, and on these a 0.0622 → 0.0582 improvement is
not separable from noise on the calibration split. `scripts/decline_rule.py`
argued that burden belongs on accepting rather than on declining, against
seven constructed heads; this is the first time the rule has run on a real
corpus, and it declined three times without costing a gate.

That is the intended behaviour and it is worth stating plainly rather than
celebrating: the model is well enough calibrated on this corpus that post-hoc
correction cannot prove it helps. A model that needed the calibrator would
have got it — seed 2 did, 0.0642 → 0.0229.

**And on seed 2 the calibrator cost accuracy.** Uncalibrated it scores 0.7194
at ECE 0.0335; calibrated it scores 0.7126 at ECE 0.0185. An isotonic map is
monotone per class and not jointly, so it can reorder two classes and move the
argmax — 0.68 accuracy points for nearly halving the calibration error.

That is a real trade and this repository's first ground rule is never to make
one without measuring both sides. It was being made invisibly: the report's
per-question table read the *uncalibrated* run while the gate above it read
the calibrated one, and printed 0.7194 and 0.7126 three sections apart with
neither labelled. Nobody noticed for as long as calibration never changed a
decision, which on the synthetic corpus it never did. Fixed, and the table now
names the suite it came from.

## What this is not

**Not state of the art.** Banking77 is a solved benchmark; a fine-tuned BERT
reaches the low 90s. This is a 128-wide, two-layer model trained from scratch
on 7,083 examples with a byte-level BPE built from the project's own corpus.
72–74% is a real number for that, and it is not a claim about the ceiling.

**Not five data streams.** One corpus, one primitive. HelpSteer2 loads and has
never been trained on; the annotator-distribution data — the stream that
teaches a model what disagreement looks like — is still unbuilt.

**Not transferable to the synthetic findings.** `size` is still unlearned and
the reference model is still a spike. What this shows is that the pipeline —
compiler, trainer, calibrator selection, gates, floors — produces a certified,
well-calibrated model when it is given real data and enough of it.
