"""
Phase 2 – Kernel Tuning for RMSNorm, LayerNorm, LinearGELU, SwiGLU
=====================================================================
Benchmarks tile/block-size configurations for four kernels:

  1. rmsnorm_kernel    – BLOCK_SIZE sweep (power-of-2)
  2. layernorm_kernel  – BLOCK_SIZE sweep (power-of-2)
  3. linear_gelu_kernel – (TILE_M, TILE_N, TILE_K, warps, stages) sweep
  4. swiglu_fused_kernel – (TILE_M, TILE_N, TILE_K, warps, stages) sweep

Representative shapes are drawn from GLM-ASR inference:
  RMSNorm / LayerNorm: hidden sizes 1280, 3584, 5120  (batch 512)
  LinearGELU:  same shapes as linear_kernel_tf32 (audio encoder encoder MLP)
  SwiGLU:      decoder MLP shapes

Results printed as tables and saved to kernel_tune_results.txt.

Usage:
    python kernel_tune.py
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Import kernels from layers.py
# ---------------------------------------------------------------------------
_here = Path(__file__).parent
sys.path.insert(0, str(_here))
from layers import (  # noqa: E402
    rmsnorm_kernel,
    layernorm_kernel,
    linear_gelu_kernel,
    swiglu_fused_kernel,
)

# ============================================================================
# Shared helpers
# ============================================================================

def _pad(size: int, tile: int) -> int:
    return ((size + tile - 1) // tile) * tile


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length() if x > 0 else 1


def _print_section(title: str):
    bar = "=" * 72
    print(f"\n{bar}\n{title}\n{bar}")


def _print_summary(
    configs: list,
    shapes: List[Tuple],
    results: Dict[str, Dict[str, float]],
    shape_key_fn,      # (shape_tuple) -> str key used in results
    shape_label_fn,    # (shape_tuple) -> str for display
    cfg_label_fn,      # (cfg) -> str
    output_lines: list,
):
    """Generic summary table printer / collector."""
    col_w = 14
    shape_keys   = [shape_key_fn(s)   for s in shapes]
    shape_labels = [shape_label_fn(s) for s in shapes]

    sep = "-" * (26 + col_w * len(shapes))
    output_lines.append(sep)
    hdr = f"{'Config':<26}" + "".join(f"{lbl:>{col_w}}" for lbl in shape_labels)
    output_lines.append(hdr)
    output_lines.append(sep)

    best_per_shape = {k: (float("inf"), "–") for k in shape_keys}
    finite_sums: Dict[str, float] = {}

    for cfg in configs:
        lbl = cfg_label_fn(cfg)
        row = f"{lbl:<26}"
        total = 0.0
        for sk in shape_keys:
            ms = results.get(cfg_label_fn(cfg), {}).get(sk, float("inf"))
            cell = f"{ms:.3f}" if ms != float("inf") else "SKIP"
            row += f"{cell:>{col_w}}"
            if ms < best_per_shape[sk][0]:
                best_per_shape[sk] = (ms, lbl)
            total += ms
        finite_sums[lbl] = total
        output_lines.append(row)

    output_lines.append(sep)

    # Best-per-shape row
    best_row = f"{'BEST':<26}"
    for sk in shape_keys:
        _, name = best_per_shape[sk]
        best_row += f"{name:>{col_w}}"
    output_lines.append(best_row)
    output_lines.append(sep)

    # Sum-latency ranking
    output_lines.append("\nSum latency across all shapes (ms):")
    best_name = min(finite_sums, key=finite_sums.get)
    for lbl, total in sorted(finite_sums.items(), key=lambda kv: kv[1]):
        marker = "  ← BEST" if lbl == best_name else ""
        val = f"{total:.3f}" if total != float("inf") else "SKIP"
        output_lines.append(f"  {lbl:<28} {val}{marker}")
    output_lines.append("")


# ============================================================================
# 1 & 2 – RMSNorm / LayerNorm  (BLOCK_SIZE sweep)
# ============================================================================

# BLOCK_SIZE must be a power-of-two constexpr in the kernel.
NORM_BLOCK_SIZES = [64, 128, 256, 512, 1024, 2048, 4096]

# (batch, hidden_size, label)
NORM_SHAPES: List[Tuple[int, int, str]] = [
    (512, 1280, "B=512  H=1280"),
    (512, 3584, "B=512  H=3584"),
    (512, 5120, "B=512  H=5120"),
    (  1, 3584, "B=1    H=3584"),   # single-token decode
]


def _bench_rmsnorm(block: int, batch: int, hidden: int, device) -> float:
    if block >= hidden:   # block must be >= hidden for masking to work
        return float("inf")
    try:
        x = torch.randn((batch, hidden), dtype=torch.float32, device=device)
        w = torch.ones(hidden, dtype=torch.float32, device=device)
        y = torch.empty_like(x)

        def _run():
            rmsnorm_kernel[(batch,)](
                x, w, y,
                x.stride(0), y.stride(0),
                hidden, 1e-6,
                BLOCK_SIZE=block,
            )

        _run(); torch.cuda.synchronize()
        return triton.testing.do_bench(_run, warmup=25, rep=200)
    except Exception as exc:
        print(f"    [SKIP] RMSNorm BLOCK={block} ({batch},{hidden}): {exc}")
        return float("inf")


def _bench_layernorm(block: int, batch: int, hidden: int, device) -> float:
    if block >= hidden:
        return float("inf")
    try:
        x = torch.randn((batch, hidden), dtype=torch.float32, device=device)
        w = torch.ones(hidden, dtype=torch.float32, device=device)
        b = torch.zeros(hidden, dtype=torch.float32, device=device)
        y = torch.empty_like(x)

        def _run():
            layernorm_kernel[(batch,)](
                x, w, b, y,
                x.stride(0), y.stride(0),
                hidden, 1e-5,
                BLOCK_SIZE=block,
            )

        _run(); torch.cuda.synchronize()
        return triton.testing.do_bench(_run, warmup=25, rep=200)
    except Exception as exc:
        print(f"    [SKIP] LayerNorm BLOCK={block} ({batch},{hidden}): {exc}")
        return float("inf")


def tune_norm_kernels(device, output_lines: list):
    """Sweep BLOCK_SIZE for rmsnorm_kernel and layernorm_kernel."""

    for kernel_name, bench_fn in [
        ("RMSNorm  (rmsnorm_kernel)",   _bench_rmsnorm),
        ("LayerNorm (layernorm_kernel)", _bench_layernorm),
    ]:
        _print_section(f"Kernel: {kernel_name}")
        output_lines.append(f"\n{'='*72}")
        output_lines.append(f"Kernel: {kernel_name}")
        output_lines.append(f"{'='*72}")

        # results[block_size_str][shape_label] = ms
        results: Dict[str, Dict[str, float]] = {}

        valid_blocks = []
        for block in NORM_BLOCK_SIZES:
            key = f"BLOCK={block}"
            results[key] = {}
            any_valid = False
            print(f"\n  BLOCK_SIZE={block}")
            for batch, hidden, label in NORM_SHAPES:
                ms = bench_fn(block, batch, hidden, device)
                results[key][label] = ms
                tag = f"{ms:.4f} ms" if ms != float("inf") else "SKIP"
                print(f"    {label}  →  {tag}")
                if ms != float("inf"):
                    any_valid = True
            if any_valid:
                valid_blocks.append(block)

        # Summary
        configs  = [f"BLOCK={b}" for b in NORM_BLOCK_SIZES]
        shapes   = NORM_SHAPES

        _print_summary(
            configs=configs,
            shapes=shapes,
            results=results,
            shape_key_fn=lambda s: s[2],
            shape_label_fn=lambda s: s[2],
            cfg_label_fn=lambda c: c,
            output_lines=output_lines,
        )

        # Print best
        sums = {
            f"BLOCK={b}": sum(results[f"BLOCK={b}"].get(s[2], float("inf"))
                              for s in NORM_SHAPES)
            for b in NORM_BLOCK_SIZES
        }
        best_key = min(sums, key=sums.get)
        best_block = int(best_key.split("=")[1])
        rec = (f"  → Set BLOCK_SIZE={best_block} in {kernel_name.split()[0].lower()}_kernel "
               f"calls  (sum={sums[best_key]:.3f} ms)")
        print(rec)
        output_lines.append(rec)


# ============================================================================
# 3 – linear_gelu_kernel  (TILE_M × TILE_N × TILE_K sweep)
# ============================================================================

@dataclass
class MatmulConfig:
    name: str
    TILE_M: int
    TILE_N: int
    TILE_K: int
    num_warps: int
    num_stages: int
    note: str = ""


# Configs tuned for linear_gelu (encoder MLP: large M, single accumulator)
LINEARGELU_CONFIGS: List[MatmulConfig] = [
    MatmulConfig("Default",    64,  64,  32, 4, 2, "baseline"),
    MatmulConfig("Tiny",       32,  32,  32, 2, 2, "low register pressure"),
    MatmulConfig("Spec-A",    128,  64,  32, 4, 3, "assignment Config A"),
    MatmulConfig("Spec-B",     64, 128,  64, 8, 3, "assignment Config B"),
    MatmulConfig("Spec-C",    128, 128,  32, 8, 4, "assignment Config C"),
    MatmulConfig("Wide-MN",   128, 128,  64, 8, 3, "large tiles all dims"),
    MatmulConfig("LargeM",    256,  64,  32, 8, 4, "very wide M"),
    MatmulConfig("LargeN",     64, 256,  32, 8, 4, "very wide N"),
    MatmulConfig("K64-w4",     64,  64,  64, 4, 2, "larger K tile"),
    MatmulConfig("K64-w8",    128,  64,  64, 8, 3, "large K, wide M"),
    MatmulConfig("S2-w2",     128, 128,  32, 2, 2, "fewer warps/stages"),
    MatmulConfig("Enc-opt",   128, 128,  32, 4, 3, "encoder sweet-spot"),
]

# encoder MLP shapes: M large (prefill), single GELU accumulator
LINEARGELU_SHAPES: List[Tuple[int, int, int, str]] = [
    (512, 1280, 5120, "M=512  K=1280 N=5120"),   # fc1 up-proj
    (512, 5120, 1280, "M=512  K=5120 N=1280"),   # fc2 down-proj (no GELU but same tile)
    (128, 1280, 5120, "M=128  K=1280 N=5120"),   # shorter audio chunk
    ( 64, 1280, 5120, "M=64   K=1280 N=5120"),   # short chunk
]


def _bench_linear_gelu(cfg: MatmulConfig, M: int, K: int, N: int, device) -> float:
    try:
        M_pad = _pad(M, cfg.TILE_M)
        K_pad = _pad(K, cfg.TILE_K)
        N_pad = _pad(N, cfg.TILE_N)

        a = torch.zeros((M_pad, K_pad), dtype=torch.float32, device=device)
        b = torch.zeros((K_pad, N_pad), dtype=torch.float32, device=device)
        c = torch.zeros((M_pad, N_pad), dtype=torch.float32, device=device)
        a[:M, :K] = torch.randn(M, K, device=device)
        b[:K, :N] = torch.randn(K, N, device=device)

        grid = (triton.cdiv(M_pad, cfg.TILE_M), triton.cdiv(N_pad, cfg.TILE_N))

        def _run():
            linear_gelu_kernel[grid](
                a, b, c,
                M_pad, N_pad, K_pad,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BLOCK_M=cfg.TILE_M,
                BLOCK_N=cfg.TILE_N,
                BLOCK_K=cfg.TILE_K,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
            )

        _run(); torch.cuda.synchronize()
        return triton.testing.do_bench(_run, warmup=25, rep=100)
    except Exception as exc:
        print(f"    [SKIP] LinearGELU {cfg.name} ({M},{K},{N}): {exc}")
        return float("inf")


def tune_linear_gelu(device, output_lines: list):
    _print_section("Kernel: linear_gelu_kernel  (Fused Linear + GELU)")
    output_lines.append(f"\n{'='*72}")
    output_lines.append("Kernel: linear_gelu_kernel  (Fused Linear + GELU)")
    output_lines.append(f"{'='*72}")

    results: Dict[str, Dict[str, float]] = {}

    for cfg in LINEARGELU_CONFIGS:
        key = f"{cfg.name}({cfg.TILE_M},{cfg.TILE_N},{cfg.TILE_K})"
        results[key] = {}
        print(f"\n  Config: {cfg.name:<10} TILE=({cfg.TILE_M},{cfg.TILE_N},{cfg.TILE_K})"
              f"  warps={cfg.num_warps}  stages={cfg.num_stages}  [{cfg.note}]")
        for M, K, N, label in LINEARGELU_SHAPES:
            ms = _bench_linear_gelu(cfg, M, K, N, device)
            results[key][label] = ms
            tag = f"{ms:.3f} ms" if ms != float("inf") else "SKIP"
            print(f"    {label}  →  {tag}")

    # Summary table
    cfg_keys = [
        f"{c.name}({c.TILE_M},{c.TILE_N},{c.TILE_K})" for c in LINEARGELU_CONFIGS
    ]
    _print_summary(
        configs=cfg_keys,
        shapes=LINEARGELU_SHAPES,
        results=results,
        shape_key_fn=lambda s: s[3],
        shape_label_fn=lambda s: s[3],
        cfg_label_fn=lambda c: c,
        output_lines=output_lines,
    )

    sums = {k: sum(results[k].get(s[3], float("inf")) for s in LINEARGELU_SHAPES)
            for k in cfg_keys}
    best_key = min(sums, key=sums.get)
    best_cfg = next(c for c in LINEARGELU_CONFIGS
                    if f"{c.name}({c.TILE_M},{c.TILE_N},{c.TILE_K})" == best_key)
    rec = (f"  → EncoderMLP.TILE_M={best_cfg.TILE_M} TILE_N={best_cfg.TILE_N} "
           f"TILE_K={best_cfg.TILE_K}  (sum={sums[best_key]:.3f} ms)")
    print(rec)
    output_lines.append(rec)


# ============================================================================
# 4 – swiglu_fused_kernel  (TILE_M × TILE_N × TILE_K sweep)
# ============================================================================
# SwiGLU has TWO accumulators (gate + up) — register pressure is 2× that of
# linear_gelu.  Explore smaller tiles to avoid register spill.

SWIGLU_CONFIGS: List[MatmulConfig] = [
    MatmulConfig("Default",   64,  64,  32, 4, 2, "baseline"),
    MatmulConfig("Tiny",      32,  32,  32, 2, 2, "min register pressure"),
    MatmulConfig("Spec-A",   128,  64,  32, 4, 3, "assignment Config A"),
    MatmulConfig("Spec-B",    64, 128,  64, 8, 3, "assignment Config B"),
    MatmulConfig("Spec-C",   128, 128,  32, 8, 4, "assignment Config C"),
    MatmulConfig("Mid-MN",    64,  64,  64, 4, 2, "larger K tile"),
    MatmulConfig("Wide-M",   128,  64,  32, 4, 2, "wide M, moderate N"),
    MatmulConfig("Wide-N",    64, 128,  32, 4, 2, "wide N, moderate M"),
    MatmulConfig("Swi-opt",   64,  64,  32, 8, 3, "more warps, 3 stages"),
    MatmulConfig("M32-N128",  32, 128,  32, 4, 2, "narrow M, wide N"),
    MatmulConfig("M32-N64",   32,  64,  32, 4, 2, "small tiles"),
    MatmulConfig("Large",    128, 128,  64, 8, 4, "large all tiles"),
]

# Decoder MLP shapes for SwiGLU
#   Qwen2-Audio / GLM-ASR: hidden=3584, intermediate=18944 (×5.28)
#   Prefill (audio prompt + ~59 tokens) and single-token decode
SWIGLU_SHAPES: List[Tuple[int, int, int, str]] = [
    ( 59, 3584, 18944, "M=59   K=3584 N=18944"),  # typical prefill
    (128, 3584, 18944, "M=128  K=3584 N=18944"),  # longer prefill
    (  1, 3584, 18944, "M=1    K=3584 N=18944"),  # autoregressive decode
    ( 59, 3584,  3584, "M=59   K=3584 N=3584 "),  # down-proj
]


def _bench_swiglu(cfg: MatmulConfig, M: int, K: int, N: int, device) -> float:
    try:
        M_pad = _pad(M, cfg.TILE_M)
        K_pad = _pad(K, cfg.TILE_K)
        N_pad = _pad(N, cfg.TILE_N)

        a        = torch.zeros((M_pad, K_pad), dtype=torch.float32, device=device)
        gate_w   = torch.zeros((K_pad, N_pad), dtype=torch.float32, device=device)
        up_w     = torch.zeros((K_pad, N_pad), dtype=torch.float32, device=device)
        out      = torch.zeros((M_pad, N_pad), dtype=torch.float32, device=device)

        a[:M, :K]     = torch.randn(M, K, device=device)
        gate_w[:K, :N] = torch.randn(K, N, device=device)
        up_w[:K, :N]   = torch.randn(K, N, device=device)

        grid = (triton.cdiv(M_pad, cfg.TILE_M), triton.cdiv(N_pad, cfg.TILE_N))

        def _run():
            swiglu_fused_kernel[grid](
                a, gate_w, up_w, out,
                M_pad, N_pad, K_pad,
                a.stride(0),      a.stride(1),
                gate_w.stride(0), gate_w.stride(1),
                up_w.stride(0),   up_w.stride(1),
                out.stride(0),    out.stride(1),
                BLOCK_M=cfg.TILE_M,
                BLOCK_N=cfg.TILE_N,
                BLOCK_K=cfg.TILE_K,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
            )

        _run(); torch.cuda.synchronize()
        return triton.testing.do_bench(_run, warmup=25, rep=100)
    except Exception as exc:
        print(f"    [SKIP] SwiGLU {cfg.name} ({M},{K},{N}): {exc}")
        return float("inf")


def tune_swiglu(device, output_lines: list):
    _print_section("Kernel: swiglu_fused_kernel  (Fused Gate + Up + SiLU)")
    output_lines.append(f"\n{'='*72}")
    output_lines.append("Kernel: swiglu_fused_kernel  (Fused Gate + Up + SiLU)")
    output_lines.append(f"{'='*72}")

    results: Dict[str, Dict[str, float]] = {}

    for cfg in SWIGLU_CONFIGS:
        key = f"{cfg.name}({cfg.TILE_M},{cfg.TILE_N},{cfg.TILE_K})"
        results[key] = {}
        print(f"\n  Config: {cfg.name:<10} TILE=({cfg.TILE_M},{cfg.TILE_N},{cfg.TILE_K})"
              f"  warps={cfg.num_warps}  stages={cfg.num_stages}  [{cfg.note}]")
        for M, K, N, label in SWIGLU_SHAPES:
            ms = _bench_swiglu(cfg, M, K, N, device)
            results[key][label] = ms
            tag = f"{ms:.3f} ms" if ms != float("inf") else "SKIP"
            print(f"    {label}  →  {tag}")

    cfg_keys = [
        f"{c.name}({c.TILE_M},{c.TILE_N},{c.TILE_K})" for c in SWIGLU_CONFIGS
    ]
    _print_summary(
        configs=cfg_keys,
        shapes=SWIGLU_SHAPES,
        results=results,
        shape_key_fn=lambda s: s[3],
        shape_label_fn=lambda s: s[3],
        cfg_label_fn=lambda c: c,
        output_lines=output_lines,
    )

    sums = {k: sum(results[k].get(s[3], float("inf")) for s in SWIGLU_SHAPES)
            for k in cfg_keys}
    best_key = min(sums, key=sums.get)
    best_cfg = next(c for c in SWIGLU_CONFIGS
                    if f"{c.name}({c.TILE_M},{c.TILE_N},{c.TILE_K})" == best_key)
    rec = (f"  → MLP.TILE_M={best_cfg.TILE_M} TILE_N={best_cfg.TILE_N} "
           f"TILE_K={best_cfg.TILE_K}  (sum={sums[best_key]:.3f} ms)")
    print(rec)
    output_lines.append(rec)


# ============================================================================
# Main
# ============================================================================

def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available – aborting.")
        sys.exit(1)

    device = torch.device("cuda")
    gpu_name   = torch.cuda.get_device_name(0)
    tri_ver    = triton.__version__
    torch_ver  = torch.__version__

    header = (
        f"GPU: {gpu_name}\n"
        f"Triton: {tri_ver}   PyTorch: {torch_ver}\n"
    )
    print(header)

    output_lines: list = [header]

    # ------------------------------------------------------------------
    # Warm-up: compile all four kernels with small dummy inputs
    # ------------------------------------------------------------------
    print("Warming up JIT compilation …", flush=True)

    # norm kernels
    _x = torch.randn((2, 64), device=device)
    _w = torch.ones(64, device=device)
    _y = torch.empty_like(_x)
    _b = torch.zeros(64, device=device)
    rmsnorm_kernel[(2,)](
        _x, _w, _y, _x.stride(0), _y.stride(0), 64, 1e-6, BLOCK_SIZE=64
    )
    layernorm_kernel[(2,)](
        _x, _w, _b, _y, _x.stride(0), _y.stride(0), 64, 1e-5, BLOCK_SIZE=64
    )

    # matmul kernels (tiny 32×32×32)
    _a = torch.zeros((32, 32), device=device)
    _bm = torch.zeros((32, 32), device=device)
    _c  = torch.zeros((32, 32), device=device)
    linear_gelu_kernel[(1, 1)](
        _a, _bm, _c, 32, 32, 32,
        _a.stride(0), _a.stride(1),
        _bm.stride(0), _bm.stride(1),
        _c.stride(0), _c.stride(1),
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=32,
    )
    _g = torch.zeros((32, 32), device=device)
    _u = torch.zeros((32, 32), device=device)
    swiglu_fused_kernel[(1, 1)](
        _a, _g, _u, _c, 32, 32, 32,
        _a.stride(0), _a.stride(1),
        _g.stride(0), _g.stride(1),
        _u.stride(0), _u.stride(1),
        _c.stride(0), _c.stride(1),
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=32,
    )
    torch.cuda.synchronize()
    print("Done.\n")

    # ------------------------------------------------------------------
    # Run all four tuning sweeps
    # ------------------------------------------------------------------
    tune_norm_kernels(device, output_lines)
    tune_linear_gelu(device, output_lines)
    tune_swiglu(device, output_lines)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_path = _here / "kernel_tune_results.txt"
    with open(out_path, "w") as fh:
        fh.write("\n".join(output_lines))
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()