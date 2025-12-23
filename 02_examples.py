# %% Tiktoken Encoding alternative ###############################################
import tiktoken
enc = tiktoken.get_encoding("gpt2")
toks = enc.encode("word: silly misspelllllled definition")
print(toks)
print(enc.decode(toks))
print(f"{enc.n_vocab=}")

# %% Self attention trick ###############################################
import torch
from torch.nn import functional as F

def example():
    a = torch.tril(torch.ones(4, 4))
    a /= torch.sum(a, dim=1, keepdim=True)
    b = torch.randint(0, 10, (4,3)).float()
    c = a @ b
    print("a")
    print(a)
    print("b")
    print(b)
    print("c is a running average of the columns of B")
    print(c)

    # we want tokens to communicate over the T dimension, and avg all up to their time.
    # x[b, t] = mean_{i<=t} x[b, i]

    B, T, C = 4, 8, 2  # batch, time, channels
    x = torch.randn(B, T, C)

    wei = torch.tril(torch.ones(T, T))
    wei = wei / wei.sum(dim=1, keepdim=True)

    # wei is T, T
    # x is B, T, C
    # the matrix multiplication broadcasts over B and turns wei into B, T, T
    # (B, T, T) @ (B, T, C) --> (B, T, C)
    xbow2 = wei @ x
    print("xbow2")
    print(xbow2.shape)


    # version 2 with softmax
    wei = torch.zeros((T, T))
    tril = torch.tril(torch.ones(T, T))
    wei = wei.masked_fill(tril == 0, float('-inf'))
    # print(wei)
    wei = F.softmax(wei, dim=-1)
    print(wei)
example()

# %% SElf attention example ###############################################
from torch import nn 

def self_attention_example():
    B, T, C = 4, 8, 32
    x = torch.randn(B, T, C)

    # data emmits keys and queries
    head_size = 16
    key = nn.Linear(C, head_size, bias=False)
    query = nn.Linear(C, head_size, bias=False)
    value = nn.Linear(C, head_size, bias=False)

    k = key(x)  # B, T, head_size
    q = query(x)  # B, T, head_size

    tril = torch.tril(torch.ones(T, T))
    # wei = torch.zeros((T, T))
    # instead of starting as zeros we do the dot product

    # (B, T, head) @ (B, head, T) --> (B, T, T)
    wei = q @ k.transpose(-2, -1) * (head_size ** -0.5)  # B, T, T

    wei = wei.masked_fill(tril == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)

    v = value(x)
    out = wei @ v  # B, T, head_size
    print(out.shape)

    print("scaling - if input values are too large, becomes 1 hot encoding")
    print(torch.softmax(torch.tensor([0.1, 0.2, 0.3]), dim=-1))
    print(torch.softmax(torch.tensor([1.0, 2.0, 3.0]), dim=-1))
    print(torch.softmax(torch.tensor([10.0, 20.0, 30.0]), dim=-1))


self_attention_example()