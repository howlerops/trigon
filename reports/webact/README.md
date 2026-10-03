# Web actions: a browser agent's step, on websites the model has never seen

`scripts/webact.py` over `trigon.evals.webact`: 600 Mind2Web steps (Deng et
al., 2023, CC BY 4.0) from 81 tasks on 36 websites, one request per step in
the shape a browser agent on a typed-decision API sends -- an `operation`
Choice and one target Choice per operation over a 30-element table. No
training set includes these websites (`mind2web-train` excludes all 36).

| System | Operation | Element | **Step success** | ECE executed (floor p95) | p50 / p95 |
| --- | ---: | ---: | ---: | --- | --- |
| A compatible hosted service (2026-10-02) | **0.758** | 0.095 | **0.050** | **0.042** (0.032) | 4.17 / 5.68 s, over the network |
| Qwen3-0.6B, the six-corpus mix, no web data | 0.590 | 0.092 | 0.038 | 0.091 (0.039) | **2.60 / 3.43 s**, laptop CPU beside a training run |
| **Qwen3-0.6B broad: the mix, both teachers and 2,264 Mind2Web steps from 35 other websites** | **0.833** | **0.320** | **0.260** | **0.039 (0.058)** | 1.94 / 2.51 s, laptop CPU |

Step success is the operation right *and* the element its own head chose
right -- what the agent would execute. Chance on the element is 1/30. Both
systems find the element about one time in ten: the element, not the
operation, is the workload's hard part, and neither system has been trained
on it.

**Trained on web steps from other websites, step success goes from 0.038 to 0.260 -- five times the hosted service -- with the executed step calibrated within its floor.** Every success is a CLICK: step success on the 84 TYPE_TEXT and 16 SELECT steps is 0.0, so the typing and selecting heads are the next thing to fix. One seed.
