"""
Modal-based training script for running parallel experiments.

Usage:
    modal run train.py::train_run --run-name my_run --n-embed 256
    modal run train.py::sweep
"""
import modal

app = modal.App("urban-dict-transformer")

# Container image with dependencies and data baked in
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "wandb", "numpy", "tiktoken", "sentencepiece")
    .add_local_file("model.py", "/root/model.py")
    .add_local_file("data.py", "/root/data.py")
    .add_local_file("data/urbandict_corpus.txt", "/data/corpus.txt")
    .add_local_file("tokenizers/bpe_500.model", "/data/bpe_500.model")
    .add_local_file("tokenizers/bpe_1000.model", "/data/bpe_1000.model")
)

# Volume for checkpoints (persists across runs)
checkpoint_volume = modal.Volume.from_name("transformer-checkpoints", create_if_missing=True)

@app.function(
    gpu="T4",
    image=image,
    secrets=[modal.Secret.from_name("wandb-secret")],
    volumes={"/checkpoints": checkpoint_volume},
    timeout=9000,  # 2.5 hours (2hr runs + 30min buffer)
)
def train_run(
    n_embed: int = 128,
    num_heads: int = 4,
    n_layers: int = 4,
    ff_expand_ratio: int = 4,
    dropout: float = 0.05,
    block_size: int = 512,  # covers 95%+ of word+definition entries
    batch_size: int = 64,
    max_seconds: int = 1800,  # 30 minutes default
    eval_interval_seconds: int = 120,
    eval_iters: int = 50,
    learning_rate: float = 3e-4,
    run_name: str = None,
    tokenizer: str = "char",  # "char" or "tiktoken"
    tie_weights: bool = False,
):
    import sys
    sys.path.insert(0, "/root")

    import os
    import time
    import math
    import torch
    import wandb
    from model import Transformer, TransformerConfig
    from data import load_char_data, load_tiktoken_data, load_sentencepiece_data

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data
    if tokenizer == "tiktoken":
        train_data, val_data, vocab_size, _encode, decode, chars_per_token = load_tiktoken_data("/data/corpus.txt")
    elif tokenizer == "bpe500":
        train_data, val_data, vocab_size, _encode, decode, chars_per_token = load_sentencepiece_data("/data/corpus.txt", "/data/bpe_500.model")
    elif tokenizer == "bpe1k":
        train_data, val_data, vocab_size, _encode, decode, chars_per_token = load_sentencepiece_data("/data/corpus.txt", "/data/bpe_1000.model")
    else:
        train_data, val_data, vocab_size, _encode, decode = load_char_data("/data/corpus.txt")
        chars_per_token = 1.0

    # Batching
    def get_batch(split):
        data_split = train_data if split == "train" else val_data
        ix = torch.randint(len(data_split) - block_size, (batch_size,))
        x = torch.stack([data_split[i : i + block_size] for i in ix])
        y = torch.stack([data_split[i + 1 : i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss(model, splits=("train", "val")):
        out = {}
        model.eval()
        for split in splits:
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
        tie_weights=tie_weights,
    )

    model = Transformer(config)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())

    # Set default run_name if not provided
    if run_name is None:
        run_name = config.default_run_name()

    print(f"[{run_name}] Model has {num_params:,} parameters")
    checkpoint_path = f"/checkpoints/{run_name}.pt"

    # Training state
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    step = 0
    elapsed_before = 0.0  # time spent in previous runs
    wandb_run_id = None

    # Try to load checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        elapsed_before = ckpt["elapsed_seconds"]
        wandb_run_id = ckpt.get("wandb_run_id")
        print(f"Resumed from step {step}, {elapsed_before:.0f}s already elapsed")

    # Compile after loading (compile doesn't save/load well)
    model = torch.compile(model)

    # Wandb setup (resume if we have a run_id)
    wandb.init(
        project="urban-dict-transformer",
        name=run_name,
        id=wandb_run_id,
        resume="allow" if wandb_run_id else None,
        config={
            **config.to_dict(),  # model config
            "batch_size": batch_size,
            "max_seconds": max_seconds,
            "learning_rate": learning_rate,
            "num_params": num_params,
            "tokenizer": tokenizer,
            "chars_per_token": chars_per_token,
        },
    )
    wandb_run_id = wandb.run.id  # save for checkpoint

    # Warmup: linearly ramp LR over first 10% of training time
    warmup_seconds = max_seconds * 0.1

    start_time = time.time()
    last_eval_time = start_time
    eval_time_total = 0.0
    checkpoint_time_total = 0.0

    while True:
        elapsed_this_run = time.time() - start_time
        total_elapsed = elapsed_before + elapsed_this_run
        remaining = max_seconds - total_elapsed
        if remaining <= 0:
            break

        # Linear warmup, then constant (based on total elapsed time)
        if total_elapsed < warmup_seconds:
            lr_scale = total_elapsed / warmup_seconds
        else:
            lr_scale = 1.0
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate * lr_scale

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1

        # Eval and checkpoint based on time
        if time.time() - last_eval_time >= eval_interval_seconds:
            last_eval_time = time.time()

            eval_start = time.time()
            losses = estimate_loss(model, splits=("train",))
            eval_time_total += time.time() - eval_start

            total_elapsed = elapsed_before + (time.time() - start_time)
            train_bpc = losses["train"] / (chars_per_token * math.log(2))
            print(f"[{run_name}] [{total_elapsed:.0f}s] step {step}: train {losses['train']:.4f} bpc {train_bpc:.4f}")
            wandb.log({
                "train_loss": losses["train"],
                "train_bpc": train_bpc,
                "step": step,
                "elapsed_seconds": total_elapsed,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "gpu_memory_gb": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
            })

            # Save checkpoint
            ckpt_start = time.time()
            torch.save({
                "model": model._orig_mod.state_dict(),  # unwrap compiled model
                "optimizer": optimizer.state_dict(),
                "step": step,
                "elapsed_seconds": total_elapsed,
                "wandb_run_id": wandb_run_id,
            }, checkpoint_path)
            checkpoint_volume.commit()
            checkpoint_time_total += time.time() - ckpt_start
            print(f"Checkpoint saved to {checkpoint_path}")

    # Final eval (includes val)
    final_losses = estimate_loss(model)
    total_elapsed = elapsed_before + (time.time() - start_time)
    train_bpc = final_losses["train"] / (chars_per_token * math.log(2))
    val_bpc = final_losses["val"] / (chars_per_token * math.log(2))
    print(f"Final: {run_name}train loss {final_losses['train']:.4f} (bpc {train_bpc:.4f}), val loss {final_losses['val']:.4f} (bpc {val_bpc:.4f})")
    wandb.log({
        "train_loss": final_losses["train"],
        "train_bpc": train_bpc,
        "val_loss": final_losses["val"],
        "val_bpc": val_bpc,
        "step": step,
    })

    # Timing breakdown
    train_time = total_elapsed - eval_time_total - checkpoint_time_total
    print(f"\nTiming breakdown:")
    print(f"  Training:     {train_time:.1f}s ({100*train_time/total_elapsed:.1f}%)")
    print(f"  Eval:         {eval_time_total:.1f}s ({100*eval_time_total/total_elapsed:.1f}%)")
    print(f"  Checkpoint:   {checkpoint_time_total:.1f}s ({100*checkpoint_time_total/total_elapsed:.1f}%)")

    # Generate sample
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = decode(model.generate(idx, max_new_tokens=200)[0].tolist())
    print(f"\nGenerated sample:\n{generated}")
    wandb.log({"generated_sample": generated})

    # Clean up checkpoint after successful completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        checkpoint_volume.commit()
        print(f"Removed checkpoint {checkpoint_path}")

    wandb.finish()

    return {
        "run_name": run_name,
        "final_train_loss": final_losses["train"],
        "final_val_loss": final_losses["val"],
        "final_train_bpc": train_bpc,
        "final_val_bpc": val_bpc,
        "num_params": num_params,
        "total_steps": step,
        "total_seconds": total_elapsed,
    }


@app.local_entrypoint()
def main():
    """Run a single training job with default config."""
    result = train_run.remote()
    print(f"Result: {result}")


@app.function(image=image, timeout=5400)  # 1.5hr buffer
def sweep():
    """Small vocab BPE tokenization sweep."""
    configs = [
        # BPE 500 vocab - block_size=112 → ~252 chars (matches char baseline)
        {"n_embed": 256, "num_heads": 8, "n_layers": 6, "block_size": 112,
         "run_name": "bpe500_match", "tokenizer": "bpe500", "tie_weights": True,
         "max_seconds": 3600, "learning_rate": 3e-3, "batch_size": 64},

        {"n_embed": 256, "num_heads": 8, "n_layers": 8, "block_size": 112,
         "run_name": "bpe500_deep", "tokenizer": "bpe500", "tie_weights": True,
         "max_seconds": 3600, "learning_rate": 3e-3, "batch_size": 64},

        {"n_embed": 320, "num_heads": 8, "n_layers": 6, "block_size": 112,
         "run_name": "bpe500_wide", "tokenizer": "bpe500", "tie_weights": True,
         "max_seconds": 3600, "learning_rate": 3e-3, "batch_size": 64},

        # BPE 1000 vocab - block_size=96 → ~253 chars (matches char baseline)
        {"n_embed": 256, "num_heads": 8, "n_layers": 6, "block_size": 96,
         "run_name": "bpe1k_match", "tokenizer": "bpe1k", "tie_weights": True,
         "max_seconds": 3600, "learning_rate": 3e-3, "batch_size": 64},

        {"n_embed": 256, "num_heads": 8, "n_layers": 8, "block_size": 96,
         "run_name": "bpe1k_deep", "tokenizer": "bpe1k", "tie_weights": True,
         "max_seconds": 3600, "learning_rate": 3e-3, "batch_size": 64},

        {"n_embed": 320, "num_heads": 8, "n_layers": 6, "block_size": 96,
         "run_name": "bpe1k_wide", "tokenizer": "bpe1k", "tie_weights": True,
         "max_seconds": 3600, "learning_rate": 3e-3, "batch_size": 64},
    ]

    # Launch all experiments in parallel
    handles = [train_run.spawn(**config) for config in configs]
    results = [h.get() for h in handles]

    print("\n" + "=" * 70)
    print("SWEEP RESULTS (sorted by val_bpc)")
    print("=" * 70)
    print("Baselines: char-level 1.75 BPC | gpt2_8m 1.66 BPC")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x["final_val_bpc"]):
        print(f"{r['run_name']:15s} | params: {r['num_params']:>10,} | steps: {r['total_steps']:>6,} | train_bpc: {r['final_train_bpc']:.4f} | val_bpc: {r['final_val_bpc']:.4f}")

    return results
