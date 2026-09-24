# HelpSteer2: how predictable the ratings are, measured from the annotators

**HelpSteer2 (Wang et al., 2024), NVIDIA. CC BY 4.0. https://huggingface.co/datasets/nvidia/HelpSteer2**

`scripts/annotator_ceiling.py`. The `disagreements/` split: 23,652 pairs with every annotator's rating, 5,818 held out by prompt hash exactly as `helpsteer2-annotators` holds them out.

## Predicting one annotator's rating (the `helpsteer2-annotators` target)

| Question | Marginal | Naive oracle | Lift | Leave-one-out oracle | **Lift** |
| --- | ---: | ---: | ---: | ---: | ---: |
| `helpfulness` | 0.4129 | 0.6641 | +0.2513 | 0.4725 | **+0.0596** |
| `correctness` | 0.4648 | 0.6894 | +0.2246 | 0.5050 | **+0.0402** |
| `coherence` | 0.6927 | 0.7607 | +0.0681 | 0.6696 | **-0.0230** |
| `complexity` | 0.4697 | 0.6810 | +0.2112 | 0.4902 | **+0.0205** |
| `verbosity` | 0.5877 | 0.7377 | +0.1501 | 0.5956 | **+0.0079** |
| pooled | 0.5255 | 0.7066 | +0.1811 | 0.5466 | **+0.0210** |

The naive oracle counts the labelling annotator's own vote, and most of its
lift is that. The leave-one-out oracle predicts from the other annotators
on the same response plus the population prior -- more information about
the rating than a model of the text has -- and its pooled lift is the
ceiling for this target.

## Reproducing an aggregated label (the `helpsteer2` target, loosely)

6,100 pairs with four or more annotators on every question. One
half-panel's rounded mean predicting the other half's.

| Question | Marginal | Half predicts half | **Lift** |
| --- | ---: | ---: | ---: |
| `helpfulness` | 0.3046 | 0.3357 | **+0.0311** |
| `correctness` | 0.3421 | 0.3475 | **+0.0054** |
| `coherence` | 0.7118 | 0.6110 | **-0.1008** |
| `complexity` | 0.5744 | 0.5085 | **-0.0659** |
| `verbosity` | 0.5139 | 0.5238 | **+0.0098** |
| pooled | 0.4894 | 0.4653 | **-0.0241** |

Half-panels of two or three are noisier than the full panels behind the
aggregated labels, so this is a loose bound: a full-panel label is more
reproducible than this, and a model could in principle beat it.
