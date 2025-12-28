"""
Transformer model architecture for character-level language modeling.
No global state - all config passed as constructor arguments.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass, asdict


@dataclass
class TransformerConfig:
    vocab_size: int
    block_size: int
    n_embed: int
    num_heads: int
    n_layers: int
    ff_expand_ratio: int
    dropout: float
    tie_weights: bool = False
    use_swiglu: bool = False  # if True, use SwiGLU instead of GELU FFN

    def default_run_name(self) -> str:
        return f"e{self.n_embed}_h{self.num_heads}_l{self.n_layers}"

    def to_dict(self) -> dict:
        return asdict(self)


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention using fused scaled_dot_product_attention"""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_size = config.n_embed // config.num_heads
        self.dropout = config.dropout

        self.qkv = nn.Linear(config.n_embed, 3 * config.n_embed, bias=False)
        self.proj = nn.Linear(config.n_embed, config.n_embed)
        self.dropout_layer = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_size)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, T, head_size)
        q, k, v = qkv[0], qkv[1], qkv[2]

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0
        )

        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)
        out = self.dropout_layer(out)
        return out


class FeedForward(nn.Module):
    """MLP with expansion and projection"""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        hidden_dim = config.ff_expand_ratio * config.n_embed
        self.net = nn.Sequential(
            nn.Linear(config.n_embed, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, config.n_embed),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class SwiGLU(nn.Module):
    """SwiGLU FFN (used in LLaMA). Uses 8/3 expansion to match GELU FFN params."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        # 8/3 multiplier keeps param count similar to 4x GELU FFN
        hidden_dim = int(config.n_embed * 8 / 3)
        self.w_gate = nn.Linear(config.n_embed, hidden_dim, bias=False)
        self.w_up = nn.Linear(config.n_embed, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, config.n_embed, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class Block(nn.Module):
    """Transformer block: attention then feedforward with residual connections"""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.ffwd = SwiGLU(config) if config.use_swiglu else FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Transformer(nn.Module):
    """Character-level transformer language model"""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embed)

        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.n_embed)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size)

        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device

        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, stop_token=None):
        """Generate tokens. If stop_token is set, stop each sequence when it's produced."""
        self.eval()
        B = idx.shape[0]
        done = torch.zeros(B, dtype=torch.bool, device=idx.device)

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Once done, keep emitting stop token (as padding)
            if stop_token is not None:
                idx_next = torch.where(done.unsqueeze(1), stop_token, idx_next)
                done = done | (idx_next.squeeze(1) == stop_token)

            idx = torch.cat((idx, idx_next), dim=1)

            # Early exit if all sequences are done
            if stop_token is not None and done.all():
                break

        self.train()
        return idx
