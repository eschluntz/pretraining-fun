"""Data loading utilities for character-level language modeling."""
import torch


def load_char_data(corpus_path: str, train_split: float = 0.95):
    """
    Load corpus and prepare for character-level training.

    Returns:
        train_data: tensor of training token ids
        val_data: tensor of validation token ids
        vocab_size: number of unique characters
        encode: function to convert string to token ids
        decode: function to convert token ids to string
    """
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = f.read()

    chars = sorted(list(set(corpus)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

    def decode(toks):
        return "".join([itos[i] for i in toks])

    data = torch.tensor(encode(corpus), dtype=torch.long)
    n = int(train_split * len(data))
    train_data = data[:n]
    val_data = data[n:]

    print(f"Vocab size: {vocab_size}, Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")

    return train_data, val_data, vocab_size, encode, decode
