# Training

Two objectives. Do not conflate them.

## Intelligence: distillation

Soft labels from a teacher ensemble across a wide domain mix, KL against the
teacher distributions. This buys **accuracy and coverage**. It does not buy
calibration, and treating it as if it did is the mistake the plan calls out:
frontier models are overconfident, so their probabilities are not calibration
targets.

## Calibration: outcome-grounded training

Proper scoring rules (log loss, Brier) against **real outcomes**. Proper means
strictly incentive-compatible: the loss-minimising prediction is the true
conditional distribution, so honesty is the optimum rather than a regulariser.

Sources, in the order they are worth having:

1. public labelled corpora reformatted into the three primitives;
2. verifiable synthetic tasks — code-checkable predicates over generated state
   (implemented: `trigon.evals.synthetic_outcome_cases`);
3. prediction-with-resolution data, where reality eventually settled it.

Published relatives to adopt rather than reinvent: RLCR (Brier-score reward on
RLVR), Rewarding Doubt (log-scoring-rule RL), and calibration-aware RL on
decision-token probabilities.

**Subjective questions have no resolvable outcome.** Their calibration target
is the annotator distribution: train against multi-annotator label
distributions, never majority votes. Without this stream the calibration claims
cover objective questions only.

## Schedule

```
distillation  →  mixed (outcome loss upweighted)  →  calibration-only anneal
              →  post-hoc temperature and conformal fitting on held-out data
```

The post-hoc layer is the cheap half and it is implemented
(`trigon.calibration`). One temperature per primitive per domain, fitted by
minimising NLL — it cannot change any argmax, so it cannot cost accuracy. Re-fit
after **any** change to the serving path.

A fit that lands on a bound is not a calibrated model: it means NLL kept
improving as the distribution flattened, so the logits carry no usable signal.
`fit_temperature` raises `CalibrationWarning` rather than returning that number
quietly.

## Auxiliary losses

Each targets a documented failure mode, and each is measured by the matching
jaggedness benchmark.

**Paraphrase consistency.** Augmented state and instruction pairs, penalising
output divergence.

**Negation and complement consistency.** Paired questions — a Noul against its
negation, a Noul against a yes/no Choice — with a soft penalty pushing
`P(claim) + P(complement)` toward 1. Apply **only to constructed true
complements**, and ablate against ECE: forcing coherence on pairs that merely
look complementary is a way to miscalibrate a model in the name of consistency.
This is a differentiator only if the ablation holds.

**Injection robustness.** Adversarial state augmentation — embedded
instructions, self-classifying text, misleading framing — with labels held at
ground truth. Scoped precisely: train and measure the *steering* case, not
detection. Robustness means judging adversarial text against the schema's
criteria, so a guardrail question about an injection must still read it as
data.

## Explicit non-goals

No arithmetic, counting or date-comparison competence targets. "Keep math in
code" is the contract, and the jaggedness suite measures those anyway so the
contract is a published number rather than a claim.

## Tiers and backbones

**Base checkpoints, never instruct.** RLHF rewards confident-sounding output
and is a known way to wreck logit calibration; instruct models are pre-damaged
for this task.

**The workhorse must be small.** ~100 ms latency at $0.042/MTok implies a small
model. An L4 at ~$0.80/hr doing ~30k prefill tok/s at high utilisation is around
$0.007/MTok for a ~2B dense model. An 80B-A3B MoE needs H100-class memory per
replica regardless of active parameters — the wrong workhorse whatever its
active-parameter count says.

| Tier | Model | Role | Hardware |
| --- | --- | --- | --- |
| Workhorse | 1–4B dense base | 90%+ of traffic | L4 / L40S |
| Premium | ~30B dense or 80B-A3B MoE base | escalation target | H100 |
| Teacher | premium + frontier ensemble | labels only, never served | offline |

Full fine-tune preferred over LoRA for the workhorse: it is small enough to
afford, and calibration objectives touch the whole distribution. LoRA is fine
for the premium-tier spike.

## What is implemented here

The post-hoc calibration layer, the metrics, the gates, the eval harness, the
verifiable synthetic data stream — and an outcome-grounded training loop that
closes the circuit:

```bash
trigon train --out reports/run.md
```

It generates labelled cases, fits the readout heads against proper scoring
rules, measures calibration on a held-out split generated from a different
seed, fits a temperature on the training split, and runs the release gates. The
result is that the gates are passed or failed by a model rather than asserted
about one, and every later change has a baseline to regress against.

Three properties of that loop are load-bearing and easy to get wrong:

**Training and serving share one forward pass.** `TorchReadoutBackend.logits`
is the only implementation; `infer` is that function under `no_grad`. A trainer
carrying its own copy is the standard way to end up with a model that scores
well offline and is miscalibrated in production, and the divergence is
invisible until someone measures ECE on the served path.
`tests/test_training.py` asserts the two agree.

**The temperature is fitted on the training split, never on the split the
gates read.** Fitting and reporting on the same data is how a calibration
number stops meaning anything.

**A run is servable, not just reportable.** `--save-model` writes the weights
alongside the config needed to rebuild them; `trigon ask --backend torch
--weights` and `trigon serve --backend torch --weights` both load it. Without
that, a calibration report describes a model that no longer exists.

"Servable" has to mean *over the API*, not only from the CLI. It did not for a
while: the gateway built its compiler with the character heuristic while the
backend built tensors from its own tokenizer, so every torch request returned a
500 — with the whole suite green, because nothing exercised that path. `Engine`
now takes the estimator from the backend, and `tests/test_server.py` asserts the
gateway and the CLI answer identically.

**A build is named after its weights.** `model_version` is the only identifier
that reaches the caller, so an untrained model must not answer under a trained
one's name. The version is stamped when training finishes — before the report is
rendered — from a hash of every parameter, so two runs are distinguishable and
the report, the checkpoint and the served response all name the same model.

**The data carries irreducible noise.** `--noise 0.2` flips a fifth of the
labels, which is what makes calibration testable at all: on a noiseless set a
model can be right every time and any confidence below 1.0 reads as
miscalibrated. With noise, the correct behaviour is a probability near `1 -
noise`, and the suite can tell a calibrated model from a merely confident one.
The Bayes-optimal loss is computable for this generator, so "how far from
optimal" is a real number rather than a vibe.

What is still phase 2: the teacher ensemble and the distillation half, the
five real data streams, the auxiliary consistency losses, and a backbone worth
the name. The reference model here is a spike — its job is to prove the
pipeline, not to be the workhorse. Padded batching is the first thing to add
when this moves to real weights; today it is one request per forward pass with
gradient accumulation, which is CPU-friendly and honest about being small.
