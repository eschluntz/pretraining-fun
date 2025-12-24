# Pretraining Experiments
Pretraining tiny models on Urban Dictionary dataset to learn about PT.

# Files
Exploration and data prep are in 00_* through 04_*.

model.py: Contains the actual transformer
train.py: modal launch script for training runs

# Experiments

### torch.compile
On a tiny model, compilation had 20s overhead and ran 20% faster. Will be worth it for all non-trivial models!

## Model Size, training speed, and final val loss

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

**Results:** TBD

