# The calibrator decline rule, at 77 classes

`scripts/decline_power.py`, run 2026-09-24 on this session's VM. It runs the
shipped per-primitive decision, `trigon.cli._choose_calibrator`, on 77-way
heads whose true calibration is known -- Banking77's regime: ~90% accuracy,
confidences clustered near 1 -- and scores each decision on 5,000 answers it
never saw, with the statistic the gates read, max(ECE, adaptive ECE).

**Why it was run.** The certified Banking77 model (`reports/banking77/`) came
out of training overconfident by ~5 points on every seed. On two seeds the
isotonic map took that to 0.01-0.02; on the other two the rule declined a map
whose held-out check read 0.0486 -> 0.0261 and 0.0420 -> 0.0318, and the
model shipped raw at 0.0489 and 0.0448 -- 0.001 and 0.005 under a 0.05 gate.
One seed flipped between the two outcomes across two runs.

## 77 classes

```
77-way heads, 20 trials per cell, scored on 5000 fresh answers

head                 rule          applied     mean    worst       raw
------------------------------------------------------------------------
calibrated           shipped         4/20    0.0106   0.0197    0.0100
calibrated           more data       1/20    0.0107   0.0198  
calibrated           looser         12/20    0.0141   0.0259  
calibrated           unless harm    20/20    0.0158   0.0259  

overconfident 0.03   shipped        13/20    0.0258   0.0409    0.0330
overconfident 0.03   more data      10/20    0.0248   0.0354  
overconfident 0.03   looser         18/20    0.0215   0.0409  
overconfident 0.03   unless harm    20/20    0.0215   0.0389  

overconfident 0.05   shipped        14/20    0.0305   0.0604    0.0515
overconfident 0.05   more data      17/20    0.0239   0.0524  
overconfident 0.05   looser         20/20    0.0220   0.0406  
overconfident 0.05   unless harm    20/20    0.0217   0.0357  

overconfident 0.08   shipped        19/20    0.0271   0.0844    0.0803
overconfident 0.08   more data      20/20    0.0207   0.0316  
overconfident 0.08   looser         20/20    0.0241   0.0438  
overconfident 0.08   unless harm    20/20    0.0241   0.0438  

tilted               shipped        19/20    0.0237   0.0630    0.0553
tilted               more data      17/20    0.0225   0.0553  
tilted               looser         20/20    0.0225   0.0388  
tilted               unless harm    20/20    0.0232   0.0532  

exit 0
```

`raw` is the head with no calibrator at all. The worst column is the one a
release gate reads.

- **The shipped rule fails the gate on three of five shapes** -- worst
  0.0604 at the regime the real model is in, 0.0844 at 8 points over, 0.0630
  tilted. It declines the map often enough that some draws ship raw.
- **More calibration data helps and is not enough**: 2,000 answers still
  fails twice.
- **An 80% threshold is the only rule under 0.05 on every shape** -- worst
  0.0438 -- and costs a calibrated head 0.0259 against 0.0197, well inside
  the gate.
- The mirror rule (apply unless harm is shown) fails the tilted head.

## Four classes, the original study, at both thresholds

`scripts/decline_rule.py --trials 40`, temperature only, with the shipped
threshold at 0.95 and at 0.80. "strict" is the shipped rule.

At 0.95:
```
already calibrated         strict          40/40     0.0057     0.0188
already calibrated         never scale     40/40     0.0057     0.0188
slightly overconfident     strict          32/40     0.0269     0.0460
slightly overconfident     never scale     40/40     0.0306     0.0460
clearly overconfident      strict           0/40     0.0176     0.0528
clearly overconfident      never scale     40/40     0.2004     0.2145
clearly underconfident     strict           0/40     0.0184     0.0463
clearly underconfident     never scale     40/40     0.1991     0.2173
spread, calibrated         strict          40/40     0.0129     0.0208
spread, calibrated         never scale     40/40     0.0129     0.0208
spread, tilted             strict          40/40     0.0500     0.0624
spread, tilted             never scale     40/40     0.0500     0.0624
spread, tilted hard        strict          40/40     0.1079     0.1218
spread, tilted hard        never scale     40/40     0.1079     0.1218
```

At 0.80:
```
already calibrated         strict          39/40     0.0060     0.0188
already calibrated         never scale     40/40     0.0057     0.0188
slightly overconfident     strict          19/40     0.0218     0.0460
slightly overconfident     never scale     40/40     0.0306     0.0460
clearly overconfident      strict           0/40     0.0176     0.0528
clearly overconfident      never scale     40/40     0.2004     0.2145
clearly underconfident     strict           0/40     0.0184     0.0463
clearly underconfident     never scale     40/40     0.1991     0.2173
spread, calibrated         strict          39/40     0.0127     0.0208
spread, calibrated         never scale     40/40     0.0129     0.0208
spread, tilted             strict          39/40     0.0501     0.0624
spread, tilted             never scale     40/40     0.0500     0.0624
spread, tilted hard        strict          40/40     0.1079     0.1218
spread, tilted hard        never scale     40/40     0.1079     0.1218
```

**Identical worst case on all seven shapes.** The mean improves on the one
shape where the original study found the strict rule losing -- the slightly
overconfident head, 0.0269 -> 0.0218 -- and moves by 0.0003 on the others.

## On the certified model

Re-gating the four certified Banking77 checkpoints under 0.80
(`reports/banking77/README.md`): seed 0 now applies the isotonic map it had
declined, 0.0489 → 0.0202; seed 1 still declines, at 0.0448; seeds 2 and 3
are unchanged. Median ECE 0.0332 → 0.0209.

## What changed

`ACCEPT_CONFIDENCE` 0.95 -> 0.80. The burden of proof still sits on
applying a calibrator; the proof required now matches what a 500-answer
check can deliver.
