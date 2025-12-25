# Pretraining Experiments
Pretraining tiny models on Urban Dictionary dataset to learn about PT.

# Files
Exploration and data prep are in 00_* through 04_*.

model.py: Contains the actual transformer
train.py: modal launch script for training runs

# Experiments

## 1: torch.compile
On a tiny model, compilation had 20s overhead and ran 20% faster. Will be worth it for all non-trivial models!

## 2: Model Size, training speed, and final val loss

**Question:** How does model size affect training speed and final loss, given equal wall-clock training time?

**Varied params:**
| Name   | n_embed | heads | layers | ~Params |
|--------|---------|-------|--------|---------|
| tiny   | 64      | 4     | 2      | 130K    |
| small  | 128     | 4     | 4      | 850K    |
| medium | 256     | 8     | 6      | 5M      |
| large  | 384     | 12    | 8      | 14M     |
| xl     | 512     | 16    | 10     | 31M     |

**Held constant:**
- block_size=512, batch_size=64, dropout=0.2, ff_ratio=4
- lr=3e-4 with 10% warmup
- Training time: 30 min each
- GPU: T4

**Results:**
order of runs from lowest loss to highest loss:
1. medium
2. small (even though it only ran 1/2 the time, and might have been on track to beat medium),
3. tiny (looks like it's flatlined),
4. large (going down steeply!),
5. xl (loss is much higher than the rest and quite flat)

![Model Size Experiment Results](img/size_experiment.png)

## 3: Repeat of size experiment with 2 hrs

**Varied params:**
| Name   | n_embed | heads | layers | ~Params |
|--------|---------|-------|--------|---------|
| small  | 128     | 4     | 4      | 850K    |
| medium | 256     | 8     | 6      | 5M      |
| large  | 384     | 12    | 8      | 14M     |


**Results**
small initially was lowest loss, but then at 45 min medium surpassed it, and at 2 hours large almost tied small.
final losses:                                         
small: 1.38
medium: 1.27
large: 1.4

![Model Size Experiment Results](img/size_exp2.png)

I think medium is going to be my experiment workhorse.


## Learning Rate Experiment for Medium model

**Medium Model params:**
| Name   | n_embed | heads | layers | ~Params |
|--------|---------|-------|--------|---------|
| medium | 256     | 8     | 6      | 5M      |

**Held constant:**
TODO


| Run Name | Params | Steps | Val Loss |
|----------|--------|-------|----------|
| lr_med_1e-2 | 4,913,758 | 7,937 | 1.2181 |
| lr_med_6e-3 | 4,913,758 | 7,825 | 1.2132 |
| lr_med_3e-3 | 4,913,758 | 6,068 | 1.2284 |
| lr_med_1e-3 | 4,913,758 | 5,638 | 1.2674 |
| lr_med_3e-4 | 4,913,758 | 7,797 | 1.3365 |
| lr_med_1e-4 | 4,913,758 | 8,026 | 1.5870 |

**Result**
6e-3 was the winner, but barely any difference to the more standard 3e-3.

![LR experiment](img/lr_exp1.png)

**Inconsistent Step speed Mystery**
Two of the runs stepped slower, even though everything should have been identical.

![LR steps](img/lr_exp_steps.png)

Because it was the two middle learning rates that ran slower, the most likely cause is inconsistent hardware on Modal.

Investigating the slow runs on wandb, I can clearly see that they had lower `GPU Streaming Multiprocessor (SM) Clock Speed (MHz)` (smoking gun) and also different `Process Memory Available (MB)` (not important, but clear sign these runs were on different heterogeneous machines).

![LR clock rate](img/lr_clock_rate.png)

For any experiments that should be iso-compute I can just run based on number of steps, but for comparing across sizes, this will be trickier. I also checked my size experiments and there was +/- 10% clock speed, but not enough to make a serious difference.


# Future planned experiments
- Learning Rate
- LR + Batch size
- FFN vs Attention Ratio
- Depth vs Width
- Context length

For all of these, I should try doing small scale experiments and try to extrapolate what it means at a larger scale, then check those results!