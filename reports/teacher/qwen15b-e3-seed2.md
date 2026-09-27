# teacher-workflows

**Generated and labelled by Qwen2.5-7B-Instruct (Qwen Team, Alibaba Cloud), Apache-2.0, huggingface.co/Qwen/Qwen2.5-7B-Instruct at a09a35458c70. Teacher labels buy coverage, not calibration.**

Generated (state, schema) pairs labelled by a teacher model. **This report
measures imitation of the teacher, not calibration**: the labels are the
teacher's distributions, no calibrator was fitted to them, and no release
gate reads them (`docs/data.md`, *Teacher labels*).

| | |
| --- | ---: |
| Training cases | 3,860 |
| Held-out cases | 1,698 |
| Held-out questions | 5,714 |
| Epochs | 3 |
| Seed | 2 |
| Device | cuda (NVIDIA A10) |
| Model | qwen2.5-1.5b, LoRA rank 16 |

| Held out | Argmax agreement | Mean KL(teacher ‖ ·) |
| --- | ---: | ---: |
| student | 0.5296 | 0.8997 |
| lexical floor | 0.4697 | 1.0505 |
| ignores its input | 0.5270 | 0.9204 |

**One seed is one sample from a distribution nobody measured.**

## agreement with the teacher (Qwen2.5-7B-Instruct) -- NOT calibration

Every number in this section is measured against the teacher's labels,
which are a model's opinion and not an outcome. A student can match them
exactly and be exactly as overconfident as the teacher. Nothing here is a
calibration claim, and no release gate reads it.

| | Student | Ignores its input |
| --- | ---: | ---: |
| Argmax agreement with the teacher | 0.5296 | 0.5270 |
| Mean KL(teacher ‖ student), nats | 0.8997 | 0.9204 |
| Mean top-label confidence | 0.5225 | teacher: 0.9362 |

5,714 questions over 1,698 held-out cases.

| Against a label drawn from the teacher | ECE | Adaptive ECE | Floor p95 |
| --- | ---: | ---: | ---: |
| student | 0.0145 | 0.0190 | 0.0212 |

The student's confidence is indistinguishable from the teacher's at this n. That is
a statement about imitation; the teacher's own calibration against
computed truth is measured separately, on the verifiable holdout.

| Primitive | Questions | Agreement | Baseline | KL |
| --- | ---: | ---: | ---: | ---: |
| choice | 2,327 | 0.4822 | 0.4792 | 1.1292 |
| noul | 1,927 | 0.6632 | 0.6663 | 0.5395 |
| score | 1,460 | 0.4288 | 0.4192 | 1.0091 |

| Domain | Questions | Agreement | Baseline |
| --- | ---: | ---: | ---: |
| access_requests | 238 | 0.5462 | 0.5378 |
| accounts_payable | 266 | 0.4887 | 0.5113 |
| ad_compliance | 338 | 0.5799 | 0.5592 |
| clinical_intake | 312 | 0.4712 | 0.5096 |
| code_review | 290 | 0.4483 | 0.4621 |
| content_moderation | 248 | 0.5403 | 0.5202 |
| contract_review | 321 | 0.6293 | 0.5919 |
| ecommerce_returns | 264 | 0.4924 | 0.4811 |
| education | 311 | 0.5563 | 0.5627 |
| insurance_claims | 267 | 0.5543 | 0.5393 |
| it_incidents | 321 | 0.5389 | 0.5452 |
| logistics | 364 | 0.5495 | 0.5577 |
| payments_risk | 262 | 0.5115 | 0.5191 |
| product_reviews | 227 | 0.4670 | 0.4361 |
| public_services | 230 | 0.5609 | 0.5000 |
| real_estate | 295 | 0.5254 | 0.5458 |
| recruiting | 272 | 0.4926 | 0.5110 |
| sales_leads | 326 | 0.5276 | 0.5368 |
| support_triage | 255 | 0.4941 | 0.4627 |
| travel_bookings | 307 | 0.5765 | 0.5831 |
