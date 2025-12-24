"""
Modal-based training script for running parallel experiments.

Usage:
    modal run 06_train.py              # run default config
    modal run 06_train.py::sweep       # run parallel sweep
"""
import modal

app = modal.App("urban-dict-transformer")

# Container image with dependencies and data baked in
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "wandb", "numpy")
    .add_local_file("model.py", "/root/model.py")
    .add_local_file("data/urbandict_corpus.txt", "/data/corpus.txt")
)

@app.function(
    gpu="T4",
    image=image,
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=3600,
)
def train_run(
    n_embed: int = 128,
    num_heads: int = 4,
    n_layers: int = 4,
    ff_expand_ratio: int = 4,
    dropout: float = 0.2,
    block_size: int = 512,  # covers 95%+ of word+definition entries
    batch_size: int = 64,
    max_seconds: int = 1800,  # 30 minutes default
    eval_interval_seconds: int = 60,
    eval_iters: int = 50,
    learning_rate: float = 3e-4,
    run_name: str = None,
):
    import sys
    sys.path.insert(0, "/root")

    import torch
    import wandb
    from model import Transformer, TransformerConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load and prepare data
    with open("/data/corpus.txt", "r", encoding="utf-8") as f:
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
    n = int(0.95 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    print(f"Vocab size: {vocab_size}, Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")

    # Batching
    def get_batch(split):
        data_split = train_data if split == "train" else val_data
        ix = torch.randint(len(data_split) - block_size, (batch_size,))
        x = torch.stack([data_split[i : i + block_size] for i in ix])
        y = torch.stack([data_split[i + 1 : i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss(model):
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                xb, yb = get_batch(split)
                _, loss = model(xb, yb)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    # Create model
    config = TransformerConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embed=n_embed,
        num_heads=num_heads,
        n_layers=n_layers,
        ff_expand_ratio=ff_expand_ratio,
        dropout=dropout,
    )

    model = Transformer(config)
    model = model.to(device)
    model = torch.compile(model)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")

    # Wandb setup
    if run_name is None:
        run_name = f"e{n_embed}_h{num_heads}_l{n_layers}"

    wandb.init(
        project="urban-dict-transformer",
        name=run_name,
        config={
            "n_embed": n_embed,
            "num_heads": num_heads,
            "n_layers": n_layers,
            "ff_expand_ratio": ff_expand_ratio,
            "dropout": dropout,
            "block_size": block_size,
            "batch_size": batch_size,
            "max_seconds": max_seconds,
            "learning_rate": learning_rate,
            "num_params": num_params,
            "vocab_size": vocab_size,
        },
    )

    # Training loop (time-based)
    import time
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Warmup: linearly ramp LR over first 10% of training time
    warmup_seconds = max_seconds * 0.1

    start_time = time.time()
    last_eval_time = start_time
    step = 0

    while True:
        elapsed = time.time() - start_time
        if elapsed >= max_seconds:
            break

        # Linear warmup, then constant (could add cosine decay here later)
        if elapsed < warmup_seconds:
            lr_scale = elapsed / warmup_seconds
        else:
            lr_scale = 1.0
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate * lr_scale

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        step += 1

        # Eval based on time, not steps
        if time.time() - last_eval_time >= eval_interval_seconds:
            last_eval_time = time.time()
            losses = estimate_loss(model)
            elapsed = time.time() - start_time
            print(f"[{elapsed:.0f}s] step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            wandb.log({
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "step": step,
                "elapsed_seconds": elapsed,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "gpu_memory_gb": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
            })

    # Final eval
    final_losses = estimate_loss(model)
    print(f"Final: train loss {final_losses['train']:.4f}, val loss {final_losses['val']:.4f}")

    # Generate sample
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = decode(model.generate(idx, max_new_tokens=200)[0].tolist())
    print(f"\nGenerated sample:\n{generated}")
    wandb.log({"generated_sample": generated})

    wandb.finish()

    return {
        "run_name": run_name,
        "final_train_loss": final_losses["train"],
        "final_val_loss": final_losses["val"],
        "num_params": num_params,
        "total_steps": step,
    }


@app.local_entrypoint()
def main():
    """Run a single training job with default config."""
    result = train_run.remote()
    print(f"Result: {result}")


@app.function(image=image, timeout=7200)  # 2 hours for sweep coordinator
def sweep():
    """Run a parallel sweep over model sizes (all trained for same wall-clock time)."""
    configs = [
        # Scaling model size (width + depth together)
        {"n_embed": 64,  "num_heads": 4,  "n_layers": 2,  "run_name": "size_tiny"},
        {"n_embed": 128, "num_heads": 4,  "n_layers": 4,  "run_name": "size_small"},
        {"n_embed": 256, "num_heads": 8,  "n_layers": 6,  "run_name": "size_medium"},
        {"n_embed": 384, "num_heads": 12, "n_layers": 8,  "run_name": "size_large"},
        {"n_embed": 512, "num_heads": 16, "n_layers": 10, "run_name": "size_xl"},
    ]

    # Launch all experiments in parallel
    handles = [train_run.spawn(**config) for config in configs]
    results = [h.get() for h in handles]

    print("\n" + "=" * 60)
    print("SWEEP RESULTS (sorted by val_loss)")
    print("=" * 60)
    for r in sorted(results, key=lambda x: x["final_val_loss"]):
        print(f"{r['run_name']:20s} | params: {r['num_params']:>10,} | steps: {r['total_steps']:>6,} | val_loss: {r['final_val_loss']:.4f}")

    return results
