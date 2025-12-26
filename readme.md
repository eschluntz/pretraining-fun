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


## 4: Learning Rate Experiment for Medium model

**Medium Model params:**
| Name   | n_embed | heads | layers | ~Params |
|--------|---------|-------|--------|---------|
| medium | 256     | 8     | 6      | 5M      |

**Held constant:**
- block_size=512, batch_size=64, dropout=0.05, ff_ratio=4
- Training time: 60 min each
- GPU: T4


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

## 5: Batch Size + LR Scaling

**Question:** What batch size trains fastest for the medium model? Does sqrt-scaled LR work, or does batch_256 need a different LR?

**Model:** medium (5M params)

**Varied params:**
| Batch Size | Learning Rate | Notes                        | Val Loss |
|------------|---------------|------------------------------|----------|
| 64         | 3e-3          | baseline                     | 1.210    |
| 128        | 4.2e-3        | 3e-3 × √2                    | 1.210    |
| 256        | 3e-3          | no LR scaling                | 1.288    |
| 256        | 6e-3          | 3e-3 × √4 (sqrt scaling)     | 1.251    |
| 256        | 9e-3          | 1.5x sqrt scaling            | 1.228    |
| 256        | 1.2e-2        | linear scaling               | 1.243    |
| 512        | 8.5e-3        | OOM                          | -        |

**Held constant:**
- block_size=512, dropout=0.05, ff_ratio=4
- Training time: 60 min each
- GPU: T4
- Only eval on train set during training (val at end) to reduce overhead

**Results:**

batch_64 and batch_128 tied at 1.210 — doubling batch size with sqrt-scaled LR gives identical results.

batch_256's best LR was 9e-3 (between sqrt and linear scaling), but still slightly worse at 1.228.

**Conclusion:** For this 5M model, we hit diminishing returns around batch_128. Larger batches don't improve training speed — likely hitting the critical batch size. Stick with batch_64 or batch_128.

## 6: Context Length

**Question:** Can we train faster with shorter context? The current block_size=512 might be overkill for Urban Dictionary entries.

**Data analysis:**
```
Length distribution (characters per entry):
  Median: 122
  75th:   214
  90th:   370
  95th:   529

Entries fitting in each block_size:
  block_size=128: 52.5%
  block_size=256: 81.1%
  block_size=384: 90.7%
  block_size=512: 94.7%
```

**Model:** medium (5M params)

**Varied params:**
| block_size | % entries fit | Expected speedup |
|------------|---------------|------------------|
| 128        | 52.5%         | ~4x faster steps |
| 256        | 81.1%         | ~2x faster steps |
| 384        | 90.7%         | ~1.3x faster     |
| 512        | 94.7%         | baseline         |

**Held constant:**
- batch_size=64, lr=3e-3, dropout=0.05, ff_ratio=4
- Training time: 60 min each
- GPU: T4

**Note:** Current batching picks random positions in the corpus stream, not aligned to entry boundaries. Windows typically span 2+ entries. The "% entries fit" is about single-entry containment, not actual training windows.

**Results:**
| block_size | Steps | Chars/hr | Val Loss |
|------------|-------|----------|----------|
| 128        | 41,757 | 342M    | 1.2451   |
| 256        | 19,814 | 325M    | 1.2136   |
| 384        | 9,428  | 232M    | 1.2191   |
| 512        | 8,120  | 266M    | 1.2135   |

![ctx exp](img/ctx_exp.png)

**Conclusions:**
- ctx_256 and ctx_512 tie on final loss — no training speed advantage either way
- ctx_256 sees 22% more characters/hr but needs more steps to reach same loss
- ctx_128 is too aggressive — truncation hurts more than extra data helps
- Sticking with block_size=256 for future experiments (lower memory, same quality)

# Future planned experiments
[x] Learning Rate
[x] LR + Batch size
[ ] FFN vs Attention Ratio
[ ] Depth vs Width
[x] Context length
[ ] tokenization

For all of these, I should try doing small scale experiments and try to extrapolate what it means at a larger scale, then check those results!