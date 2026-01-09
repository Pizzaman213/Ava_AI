#!/usr/bin/env python3
"""
Benchmark: Ava vs nanoGPT-moe Speed Comparison

Tests the key optimizations from nanoGPT-moe:
1. torch.compile - 20-30% speedup
2. DDP gradient sync skip - 10-20% speedup for multi-GPU
3. Simple data loading with pin_memory + non_blocking

Usage:
    python benchmark_vs_nanogpt.py --framework ava
    python benchmark_vs_nanogpt.py --framework nanogpt
    python benchmark_vs_nanogpt.py --compare  # Run both and compare
"""

import argparse
import time
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def create_dummy_data(batch_size: int, seq_len: int, vocab_size: int, device: str):
    """Create dummy training data."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()
    return input_ids, labels


def benchmark_simple_moe(
    num_steps: int = 100,
    batch_size: int = 8,
    seq_len: int = 512,
    use_compile: bool = True,
    use_flash: bool = True,
):
    """
    Benchmark nanoGPT-style simple MoE (no aux losses).
    """
    print("\n" + "="*60)
    print("Benchmarking: Simple MoE (nanoGPT-style)")
    print(f"  torch.compile: {use_compile}")
    print(f"  Flash Attention: {use_flash}")
    print("="*60)

    # Simple MoE implementation (like nanoGPT-moe)
    class SimpleMLP(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.fc1 = nn.Linear(d_model, 4 * d_model, bias=False)
            self.fc2 = nn.Linear(4 * d_model, d_model, bias=False)
            self.gelu = nn.GELU()

        def forward(self, x):
            return self.fc2(self.gelu(self.fc1(x)))

    class SimpleMoE(nn.Module):
        def __init__(self, d_model, num_experts=8, top_k=2):
            super().__init__()
            self.experts = nn.ModuleList([SimpleMLP(d_model) for _ in range(num_experts)])
            self.gate = nn.Linear(d_model, num_experts, bias=False)
            self.top_k = top_k

        def forward(self, x):
            orig_shape = x.shape
            x = x.view(-1, x.shape[-1])
            scores = self.gate(x)
            weights, indices = torch.topk(scores, self.top_k, dim=-1)
            weights = weights.softmax(dim=-1)
            flat_indices = indices.view(-1)
            x = x.repeat_interleave(self.top_k, dim=0)
            y = torch.zeros_like(x)
            for i, expert in enumerate(self.experts):
                mask = flat_indices == i
                if mask.any():
                    y[mask] = expert(x[mask])
            y = (y.view(*weights.shape, -1) * weights.unsqueeze(-1)).sum(dim=1)
            return y.view(*orig_shape)

    class SimpleTransformerBlock(nn.Module):
        def __init__(self, d_model, n_heads):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            self.moe = SimpleMoE(d_model)

        def forward(self, x):
            x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
            x = x + self.moe(self.ln2(x))
            return x

    class SimpleModel(nn.Module):
        def __init__(self, vocab_size, d_model, n_layers, n_heads):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.pos_embed = nn.Embedding(2048, d_model)
            self.blocks = nn.ModuleList([SimpleTransformerBlock(d_model, n_heads) for _ in range(n_layers)])
            self.ln_f = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            # Weight tying
            self.embed.weight = self.lm_head.weight

        def forward(self, x, targets=None):
            B, T = x.shape
            pos = torch.arange(T, device=x.device)
            x = self.embed(x) + self.pos_embed(pos)
            for block in self.blocks:
                x = block(x)
            x = self.ln_f(x)
            logits = self.lm_head(x)
            if targets is not None:
                loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                return logits, loss
            return logits, None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 50304  # nanoGPT default

    # Create model
    model = SimpleModel(vocab_size=vocab_size, d_model=768, n_layers=6, n_heads=12).to(device)

    if use_compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), fused=True)

    # Warmup
    print("Warming up...")
    for _ in range(10):
        x, y = create_dummy_data(batch_size, seq_len, vocab_size, device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()

    for step in range(num_steps):
        x, y = create_dummy_data(batch_size, seq_len, vocab_size, device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tokens_per_sec = (num_steps * batch_size * seq_len) / elapsed
    ms_per_step = (elapsed / num_steps) * 1000

    print(f"\nResults:")
    print(f"  Steps: {num_steps}")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Time per step: {ms_per_step:.2f}ms")
    print(f"  Tokens/sec: {tokens_per_sec:,.0f}")

    return ms_per_step, tokens_per_sec


def benchmark_ava_moe(
    num_steps: int = 100,
    batch_size: int = 8,
    seq_len: int = 512,
    use_compile: bool = True,
):
    """
    Benchmark Ava MoE with full features.
    """
    print("\n" + "="*60)
    print("Benchmarking: Ava MoE (full features)")
    print(f"  torch.compile: {use_compile}")
    print("="*60)

    try:
        from ava.models.moe import EnhancedMoEModel, EnhancedMoEConfig
    except ImportError as e:
        print(f"Could not import Ava models: {e}")
        return None, None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 50304

    # Create Ava model with minimal config
    config = EnhancedMoEConfig(
        vocab_size=vocab_size,
        hidden_size=768,
        num_layers=6,
        num_attention_heads=12,
        intermediate_size=3072,
        num_experts=8,
        num_experts_per_token=2,
        max_position_embeddings=2048,
        use_flash_attention=True,
        router_z_loss_coef=0.0,  # Disable aux losses for fair comparison
        load_balance_loss_coef=0.0,
        diversity_loss_coef=0.0,
    )
    model = EnhancedMoEModel(config).to(device)

    if use_compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), fused=True)

    # Warmup
    print("Warming up...")
    for _ in range(10):
        x, y = create_dummy_data(batch_size, seq_len, vocab_size, device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(input_ids=x, labels=y)
            loss = outputs['loss']
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()

    for step in range(num_steps):
        x, y = create_dummy_data(batch_size, seq_len, vocab_size, device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(input_ids=x, labels=y)
            loss = outputs['loss']
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tokens_per_sec = (num_steps * batch_size * seq_len) / elapsed
    ms_per_step = (elapsed / num_steps) * 1000

    print(f"\nResults:")
    print(f"  Steps: {num_steps}")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Time per step: {ms_per_step:.2f}ms")
    print(f"  Tokens/sec: {tokens_per_sec:,.0f}")

    return ms_per_step, tokens_per_sec


def main():
    parser = argparse.ArgumentParser(description="Benchmark Ava vs nanoGPT-moe")
    parser.add_argument("--framework", choices=["ava", "nanogpt", "both"], default="both")
    parser.add_argument("--steps", type=int, default=50, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    args = parser.parse_args()

    print("="*60)
    print("Ava vs nanoGPT-moe Speed Benchmark")
    print("="*60)
    print(f"Configuration:")
    print(f"  Steps: {args.steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  torch.compile: {not args.no_compile}")

    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  CUDA: {torch.version.cuda}")
    else:
        print("  WARNING: Running on CPU!")

    results = {}

    if args.framework in ["nanogpt", "both"]:
        ms, tps = benchmark_simple_moe(
            num_steps=args.steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            use_compile=not args.no_compile,
        )
        results["nanogpt"] = {"ms_per_step": ms, "tokens_per_sec": tps}

    if args.framework in ["ava", "both"]:
        ms, tps = benchmark_ava_moe(
            num_steps=args.steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            use_compile=not args.no_compile,
        )
        results["ava"] = {"ms_per_step": ms, "tokens_per_sec": tps}

    # Print comparison
    if len(results) == 2 and results.get("ava") and results.get("nanogpt"):
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        nano_ms = results["nanogpt"]["ms_per_step"]
        ava_ms = results["ava"]["ms_per_step"]
        nano_tps = results["nanogpt"]["tokens_per_sec"]
        ava_tps = results["ava"]["tokens_per_sec"]

        print(f"\nTime per step:")
        print(f"  nanoGPT-style: {nano_ms:.2f}ms")
        print(f"  Ava:           {ava_ms:.2f}ms")
        if ava_ms and nano_ms:
            diff_pct = ((ava_ms - nano_ms) / nano_ms) * 100
            if diff_pct > 0:
                print(f"  Ava is {diff_pct:.1f}% slower")
            else:
                print(f"  Ava is {-diff_pct:.1f}% faster")

        print(f"\nTokens per second:")
        print(f"  nanoGPT-style: {nano_tps:,.0f}")
        print(f"  Ava:           {ava_tps:,.0f}")


if __name__ == "__main__":
    main()
