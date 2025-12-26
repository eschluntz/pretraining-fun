"""Train small BPE tokenizers and measure compression ratios."""

import sentencepiece as spm
import os
from pathlib import Path

CORPUS_PATH = "data/urbandict_corpus.txt"
MODEL_DIR = Path("tokenizers")
MODEL_DIR.mkdir(exist_ok=True)

# Get corpus stats
with open(CORPUS_PATH, "r") as f:
    corpus = f.read()
num_chars = len(corpus)
print(f"Corpus: {num_chars:,} chars")
print()

vocab_sizes = [500, 1000, 2000, 4000]

results = []
for vocab_size in vocab_sizes:
    prefix = MODEL_DIR / f"bpe_{vocab_size}"

    # Train tokenizer
    spm.SentencePieceTrainer.train(
        input=CORPUS_PATH,
        model_prefix=str(prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        num_threads=8,
    )

    # Load and encode
    sp = spm.SentencePieceProcessor(model_file=str(prefix) + ".model")
    tokens = sp.encode(corpus)
    num_tokens = len(tokens)

    chars_per_token = num_chars / num_tokens
    results.append((vocab_size, num_tokens, chars_per_token))

    print(f"vocab={vocab_size:,}: {num_tokens:,} tokens, {chars_per_token:.2f} chars/token")

# Summary table
print()
print("=" * 50)
print(f"{'Vocab':>8} | {'Tokens':>12} | {'Chars/Token':>12} | {'vs Char':>8}")
print("-" * 50)
for vocab_size, num_tokens, cpt in results:
    compression = num_chars / num_tokens
    print(f"{vocab_size:>8,} | {num_tokens:>12,} | {cpt:>12.2f} | {compression:>7.2f}x")

# Add GPT-2 for comparison
import tiktoken
enc = tiktoken.get_encoding("gpt2")
gpt2_tokens = enc.encode(corpus)
gpt2_cpt = num_chars / len(gpt2_tokens)
print(f"{'50257':>8} | {len(gpt2_tokens):>12,} | {gpt2_cpt:>12.2f} | {gpt2_cpt:>7.2f}x")
print("=" * 50)
