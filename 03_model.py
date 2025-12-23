# %% imports and constants ###############################################
import os
from duckdb import query
import torch
import torch.nn as nn
from torch.nn import functional as F
CORPUS_PATH = "data/urbandict_corpus.txt"

# %% Load data ###############################################
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    corpus = f.read()

chars = sorted(list(set(corpus)))
vocab_size = len(chars)
print(f"Unique characters:\n{''.join(chars)}")
print(f"Number of unique characters: {vocab_size}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s): return [stoi[c] for c in s]
def decode(toks): return "".join([itos[i] for i in toks])

data = torch.tensor(encode(corpus), dtype=torch.long)
print(f"Data tensor shape: {data.shape}, dtype: {data.dtype}")

n = int(0.95 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"Train data shape: {train_data.shape}, val data shape: {val_data.shape}")

# %% Model ################################################
# define model and hypers
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 16  # how many sequences will process in parallel
block_size = 8  # what is the maximum context length for predictions
eval_interval = 1000
eval_iters = 200
n_embed = 32
num_heads = 4


def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1: i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)


class Head(nn.Module):
    """One head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)  # B, T, head_size
        q = self.query(x)  # B, T, head_size

        # compute attention
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B,T,C) @ (B,C,T) --> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        # weighted aggregation
        v = self.value(x)
        out = wei @ v  # (B, T, T) @ (B, T, head_size) --> (B, T, head_size)
        return out


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embed)
        self.position_embedding_table = nn.Embedding(num_embeddings=block_size, embedding_dim=n_embed)
        self.sa_heads = MultiHeadAttention(num_heads=num_heads, head_size=n_embed // num_heads)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (Batch, Token, n_embed)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (Token, n_embed)
        x = tok_emb + pos_emb  # (Batch, Token, n_embed) broadcasting addition
        
        x = self.sa_heads(x)  # (Batch, Token, n_embed)
        logits = self.lm_head(x)  # (Batch, Token, Vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):

            idx_cond = idx[:, -block_size:]

            logits, loss = self(idx_cond)

            # focus on last Token step
            logits = logits[:, -1, :]  # becomes (B, C)

            # get tokens
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # append to sequence
        return idx
    
    def full_generate(self):
        idx = torch.zeros((1, 1), dtype=torch.long)  # starting token
        return decode(self.generate(idx, max_new_tokens=100)[0].tolist())

m = BigramLanguageModel()
m = m.to(device)

# %% Training loop ###############################################
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for step in range(10_000):
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % eval_interval == 0:
        losses = estimate_loss(m)
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

# %% Test generation ###############################################
print(m.full_generate())