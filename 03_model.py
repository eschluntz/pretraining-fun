# %% imports and constants ###############################################
import os
from duckdb import query
import torch
import torch.nn as nn
from torch.nn import functional as F
import wandb
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
n_layers = 3
ff_expand_ratio = 4
dropout = 0.2
max_steps = 10_001

run_name = f"non-compiled_r2_{n_layers}"


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
    """Multiple heads of self-attention using fused scaled_dot_product_attention"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size

        # Single fused projection for Q, K, V across all heads
        # Instead of num_heads separate (n_embed -> head_size) projections,
        # we do one big (n_embed -> 3 * num_heads * head_size) projection
        self.qkv = nn.Linear(n_embed, 3 * num_heads * head_size, bias=False)
        self.proj = nn.Linear(num_heads * head_size, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape  # (Batch=16, Time=8, Channels=32)

        # Project to Q, K, V for all heads in one matmul
        qkv = self.qkv(x)
        # x:   (B, T, n_embed)           e.g. (16, 8, 32)
        # qkv: (B, T, 3 * num_heads * head_size)  e.g. (16, 8, 96) if 4 heads * 8 head_size * 3

        # Reshape to separate out the 3 (for Q,K,V), num_heads, and head_size
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_size)
        # qkv: (B, T, 3, num_heads, head_size)  e.g. (16, 8, 3, 4, 8)

        # Permute to get (3, B, num_heads, T, head_size) so we can unpack Q, K, V
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # qkv: (3, B, num_heads, T, head_size)  e.g. (3, 16, 4, 8, 8)

        # Unpack into separate Q, K, V tensors
        q, k, v = qkv[0], qkv[1], qkv[2]
        # q, k, v each: (B, num_heads, T, head_size)  e.g. (16, 4, 8, 8)

        # Flash attention! This fused kernel does:
        #   1. Q @ K.T / sqrt(head_size)
        #   2. Causal masking
        #   3. Softmax
        #   4. Dropout (if training)
        #   5. Attention @ V
        # All in one fused CUDA kernel (when available)
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=dropout if self.training else 0.0
        )
        # out: (B, num_heads, T, head_size)  e.g. (16, 4, 8, 8)

        # Transpose to bring T back to position 1, then flatten heads
        out = out.transpose(1, 2)
        # out: (B, T, num_heads, head_size)  e.g. (16, 8, 4, 8)

        out = out.reshape(B, T, self.num_heads * self.head_size)
        # out: (B, T, num_heads * head_size)  e.g. (16, 8, 32)

        # Final projection back to n_embed (mixes information across heads)
        out = self.proj(out)
        # out: (B, T, n_embed)  e.g. (16, 8, 32)

        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """A simple feedforward layer"""

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, ff_expand_ratio * n_embed),
            nn.ReLU(),
            nn.Linear(ff_expand_ratio * n_embed, n_embed),  # projection back to n_embed
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: attention then feedforward"""

    def __init__(self, n_embed, num_heads):
        super().__init__()
        head_size = n_embed // num_heads
        self.sa_heads = MultiHeadAttention(num_heads=num_heads, head_size=head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embed)
        self.position_embedding_table = nn.Embedding(num_embeddings=block_size, embedding_dim=n_embed)
        
        self.blocks = nn.Sequential(
            *[Block(n_embed, num_heads=num_heads) for _ in range(n_layers)],
        )
        self.layer_norm_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (Batch, Token, n_embed)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (Token, n_embed)
        x = tok_emb + pos_emb  # (Batch, Token, n_embed) broadcasting addition
        
        x = self.blocks(x)  # (Batch, Token, n_embed)
        x = self.layer_norm_f(x)  # (Batch, Token, n_embed)
        logits = self.lm_head(x)  # (Batch, Token, Vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        self.eval()
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
        
        self.train()
        return idx
    
    def full_generate(self):
        idx = torch.zeros((1, 1), dtype=torch.long)  # starting token
        return decode(self.generate(idx, max_new_tokens=100)[0].tolist())


# %% Training loop ###############################################
m = BigramLanguageModel()
m = m.to(device)
m = torch.compile(m)

# Initialize wandb
wandb.init(
    project="urban-dict-transformer",
    name=run_name,
    config={
        "batch_size": batch_size,
        "block_size": block_size,
        "n_embed": n_embed,
        "num_heads": num_heads,
        "n_layers": n_layers,
        "ff_expand_ratio": ff_expand_ratio,
        "dropout": dropout,
        "max_steps": max_steps,
        "num_params": sum(p.numel() for p in m.parameters())
    }
)

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for step in range(max_steps):
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
        wandb.log({
            "train_loss": losses['train'],
            "val_loss": losses['val'],
            "step": step,
        })

wandb.finish()

# %% Test generation ###############################################
print(m.full_generate())