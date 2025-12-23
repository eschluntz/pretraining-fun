# %% Load dataset ###############################################
from datasets import load_dataset

dataset = load_dataset("csv", data_files="data/urbandict-word-defs.csv", engine='python', on_bad_lines='skip')["train"]
dataset = dataset.filter(lambda x: x["word"] is not None and x["definition"] is not None)
dataset = dataset.shuffle(seed=42)

# %% plot statistics ###############################################
import matplotlib.pyplot as plt

SAMPLE_FRAC = 0.01
sample = dataset.select(range(int(len(dataset) * SAMPLE_FRAC)))

word_lengths = [len(w) for w in sample["word"]]
def_lengths = [len(d) for d in sample["definition"]]
up_votes = sample["up_votes"]
down_votes = sample["down_votes"]

# fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# axes[0, 0].hist(word_lengths, bins=50)
# axes[0, 0].set_title("Word Length Distribution")
# axes[0, 0].set_xlabel("Characters")

# axes[0, 1].hist(def_lengths, bins=50)
# axes[0, 1].set_title("Definition Length Distribution")
# axes[0, 1].set_xlabel("Characters")

# axes[1, 0].hist(up_votes, bins=50)
# axes[1, 0].set_title("Up Votes Distribution")
# axes[1, 0].set_xlabel("Votes")

# axes[1, 1].hist(down_votes, bins=50)
# axes[1, 1].set_title("Down Votes Distribution")
# axes[1, 1].set_xlabel("Votes")

# plt.tight_layout()
# plt.show()

print(f"Word length: mean={sum(word_lengths)/len(word_lengths):.1f}, max={max(word_lengths)}")
print(f"Def length: mean={sum(def_lengths)/len(def_lengths):.1f}, max={max(def_lengths)}")
print(f"Up votes: mean={sum(up_votes)/len(up_votes):.1f}, max={max(up_votes)}")
print(f"Down votes: mean={sum(down_votes)/len(down_votes):.1f}, max={max(down_votes)}")

# %% combine dataset and write to corpus file ###############################################
import os

CORPUS_PATH = "data/urbandict_corpus.txt"

if os.path.exists(CORPUS_PATH):
    print(f"Corpus already exists at {CORPUS_PATH}, loading")
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = f.read()
else:
    corpus = "\n".join(f"{w}: {d}" for w, d in zip(dataset["word"], dataset["definition"]))
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        f.write(corpus)
    print(f"Wrote corpus to {CORPUS_PATH}")

# print the first 5 lines of the corpus
print("\n## First 5 lines of the corpus:")
for line in corpus.split("\n")[:5]:
    print(line)

# %% Char level Tokenization ###############################################
chars = sorted(list(set(corpus)))
vocab_size = len(chars)
print(f"Unique characters:\n{''.join(chars)}")
print(f"Number of unique characters: {vocab_size}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s): return [stoi[c] for c in s]
def decode(toks): return "".join([itos[i] for i in toks])

print(f"{encode('\n')=}")