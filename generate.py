"""
Load a checkpoint and generate samples.

Usage:
    python generate.py --checkpoint checkpoints/scale_50m.pt --prompt "yeet:"
    python generate.py --checkpoint checkpoints/scale_100m.pt --num-samples 5
"""
import argparse
import torch
import tiktoken
from model import Transformer, TransformerConfig


def load_model(checkpoint_path: str, device: str = "cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct config from checkpoint (need to infer from state dict)
    state_dict = ckpt["model"]

    # Infer config from state dict shapes
    vocab_size, n_embed = state_dict["token_embedding.weight"].shape
    block_size = state_dict["position_embedding.weight"].shape[0]
    n_layers = max(int(k.split(".")[1]) for k in state_dict if k.startswith("blocks.")) + 1

    # Check for SwiGLU vs GELU by looking for w_gate
    use_swiglu = any("w_gate" in k for k in state_dict)

    # Infer num_heads from qkv weight shape
    qkv_weight = state_dict["blocks.0.attn.qkv.weight"]
    # qkv shape is (3 * n_embed, n_embed), and we assume head_dim=64
    num_heads = n_embed // 64

    # Infer ff_expand_ratio (only matters if not swiglu)
    if not use_swiglu:
        ff_hidden = state_dict["blocks.0.ffwd.net.0.weight"].shape[0]
        ff_expand_ratio = ff_hidden // n_embed
    else:
        ff_expand_ratio = 4  # doesn't matter for swiglu

    config = TransformerConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embed=n_embed,
        num_heads=num_heads,
        n_layers=n_layers,
        ff_expand_ratio=ff_expand_ratio,
        dropout=0.0,  # no dropout for inference
        tie_weights=True,
        use_swiglu=use_swiglu,
    )

    model = Transformer(config)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"Loaded model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"  n_embed={n_embed}, n_layers={n_layers}, num_heads={num_heads}")
    print(f"  block_size={block_size}, use_swiglu={use_swiglu}")

    return model, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    model, config = load_model(args.checkpoint, args.device)

    enc = tiktoken.get_encoding("gpt2")

    # Newline is the entry separator in the corpus — use it as start token
    newline_token = enc.encode("\n")[0]  # token 198 in GPT-2

    for i in range(args.num_samples):
        if args.prompt:
            # Prepend newline so model sees "start of new entry" + prompt
            tokens = [newline_token] + enc.encode(args.prompt)
            idx = torch.tensor([tokens], dtype=torch.long, device=args.device)
        else:
            idx = torch.tensor([[newline_token]], dtype=torch.long, device=args.device)

        generated = model.generate(idx, max_new_tokens=args.max_tokens)
        text = enc.decode(generated[0].tolist())

        print(f"\n{'='*60}")
        if args.num_samples > 1:
            print(f"Sample {i+1}:")
        print(text)


if __name__ == "__main__":
    main()
