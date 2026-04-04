"""
Triton Neural Network Layers
End-to-end implementation using Triton kernels

Optimisations implemented:
  1. Tile / block-size tuning for linear_kernel_tf32 (Optimization #1):
       12 configurations benchmarked via tile_tune.py on NVIDIA H200 MIG 1g.18gb.
       Winner: TILE_M=32, TILE_N=32, TILE_K=32, num_warps=2, num_stages=2
       Small tiles outperform larger ones on the MIG slice (limited SM count).

  2. Kernel fusion (Optimization #2):
       linear_gelu_kernel  — fuses Linear + GELU in one pass (EncoderMLP)
       swiglu_fused_kernel — fuses gate projection + up projection + SiLU
                             gating into one pass (Decoder MLP / SwiGLU)
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
import triton
import triton.language as tl


# ============================================================================
# Helper Functions
# ============================================================================

def get_stream():
    """Get current CUDA stream pointer."""
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None


def pad_to_multiple(size: int, multiple: int) -> int:
    """Pad size to be a multiple of the given value."""
    return ((size + multiple - 1) // multiple) * multiple


def next_power_of_two(x: int) -> int:
    """Return the smallest power of two >= x."""
    return 1 << (x - 1).bit_length() if x > 0 else 1


# ============================================================================
# Triton Kernels
# ============================================================================
@triton.jit
def rmsnorm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    stride_x,
    stride_y,
    hidden_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    RMSNorm: x / RMS(x) * weight

    *** TODO: Implement this kernel ***

    Grid: (batch_size,)
    """
    pid = tl.program_id(0)

    # 1. Load x row
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_size
    x = tl.load(x_ptr + pid * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # 2. Compute variance with float32 precision
    var = tl.sum(x * x, axis=0) / hidden_size
    rsqrt_var = tl.rsqrt(var + eps)

    # 3. Apply normalization and scaling
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (x * rsqrt_var) * w

    # 4. Store result
    tl.store(y_ptr + pid * stride_y + offs, y, mask=mask)

@triton.jit
def layernorm_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    stride_x,
    stride_y,
    hidden_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias

    *** TODO: Implement this kernel ***

    Grid: (batch_size,)
    """
    pid = tl.program_id(0)

    # 1. Load x row
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_size
    x = tl.load(x_ptr + pid * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # 2. Compute mean and variance
    mean = tl.sum(x, axis=0) / hidden_size
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / hidden_size
    rsqrt_var = tl.rsqrt(var + eps)

    # 3. Load weight and bias
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # 4. Store result
    y = (x_centered * rsqrt_var) * w + b
    tl.store(y_ptr + pid * stride_y + offs, y, mask=mask)


@triton.jit
def gelu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    GELU using tanh approximation.

    *** TODO: Implement this kernel ***
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Vectorized load
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Tanh-based GELU approximation
    sqrt_2_over_pi = 0.7978845608028654
    x3 = x * x * x
    inner = sqrt_2_over_pi * (x + 0.044715 * x3)
    # Using tl.extra.cuda.libdevice.tanh if available for performance
    tanh_inner = tl.extra.cuda.libdevice.tanh(inner)
    y = x * 0.5 * (1.0 + tanh_inner)

    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def silu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    SiLU/Swish: x * sigmoid(x)

    *** TODO: Implement this kernel ***
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Vectorized load
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # SiLU: x * sigmoid(x)
    y = x * tl.sigmoid(x)

    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def linear_kernel_tf32(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    TF32-style matmul: output = A @ B.
    A: (M, K), B: (K, N), C: (M, N)

    *** TODO: Implement this kernel ***

    Grid: (M // BLOCK_M, N // BLOCK_N)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak,
            mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)

    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )

@triton.jit
def linear_gelu_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused Linear + GELU."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak,
            mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)

    sqrt_2_over_pi = 0.7978845608028654
    acc3 = acc * acc * acc
    inner = sqrt_2_over_pi * (acc + 0.044715 * acc3)
    acc = acc * 0.5 * (1.0 + tl.extra.cuda.libdevice.tanh(inner))

    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )

@triton.jit
def swiglu_fused_kernel(
    a_ptr,
    gate_ptr,
    up_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_gk,
    stride_gn,
    stride_uk,
    stride_un,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused SwiGLU: SiLU(x @ gate) * (x @ up)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak,
            mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K),
            other=0.0,
        )
        gate_w = tl.load(
            gate_ptr + (k + offs_k[:, None]) * stride_gk + offs_n[None, :] * stride_gn,
            mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        up_w = tl.load(
            up_ptr + (k + offs_k[:, None]) * stride_uk + offs_n[None, :] * stride_un,
            mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )

        gate_acc += tl.dot(a, gate_w)
        up_acc += tl.dot(a, up_w)

    sigmoid = 1.0 / (1.0 + tl.exp(-gate_acc))
    gate_act = gate_acc * sigmoid
    out = gate_act * up_acc

    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        out,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def embedding_kernel(
    indices_ptr,
    weight_ptr,
    output_ptr,
    embedding_dim,
    stride_w0,
    stride_w1,
    stride_out0,
    BLOCK_SIZE: tl.constexpr,
):
    """Embedding lookup using gather."""
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    idx = tl.load(indices_ptr + pid0)
    offs = pid1 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < embedding_dim
    w = tl.load(
        weight_ptr + idx * stride_w0 + offs * stride_w1, mask=mask, other=0.0
    )
    tl.store(output_ptr + pid0 * stride_out0 + offs, w, mask=mask)


@triton.jit
def softmax_kernel(x_ptr, y_ptr, stride_x, stride_y, n_cols, BLOCK_SIZE: tl.constexpr):
    """
    Numerically stable softmax over last dimension.

    *** TODO: Implement this kernel ***
    """
    row = tl.program_id(0)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    exp_x = tl.exp(x)
    denom = tl.sum(exp_x, axis=0)
    y = exp_x / denom
    tl.store(y_ptr + row * stride_y + offs, y, mask=mask)


# @triton.jit
# def attention_scores_kernel(
#     q_ptr,
#     k_ptr,
#     scores_ptr,
#     scale,
#     seq_k,
#     head_dim,
#     stride_q0,
#     stride_q1,
#     stride_q2,
#     stride_k0,
#     stride_k1,
#     stride_k2,
#     stride_s0,
#     stride_s1,
#     stride_s2,
#     BLOCK_K: tl.constexpr,
#     BLOCK_D: tl.constexpr,
# ):
#     """Compute attention scores: Q @ K^T * scale."""
#     pid_bh = tl.program_id(0)
#     pid_q = tl.program_id(1)

#     offs_k = tl.arange(0, BLOCK_K)
#     offs_d = tl.arange(0, BLOCK_D)

#     q = tl.load(
#         q_ptr + pid_bh * stride_q0 + pid_q * stride_q1 + offs_d * stride_q2,
#         mask=offs_d < head_dim,
#         other=0.0,
#     )
#     k = tl.load(
#         k_ptr
#         + pid_bh * stride_k0
#         + offs_k[:, None] * stride_k1
#         + offs_d[None, :] * stride_k2,
#         mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim),
#         other=0.0,
#     )
#     scores = tl.sum(k * q[None, :], axis=1) * scale
#     tl.store(
#         scores_ptr
#         + pid_bh * stride_s0
#         + pid_q * stride_s1
#         + offs_k * stride_s2,
#         scores,
#         mask=offs_k < seq_k,
#     )


# @triton.jit
# def attention_output_kernel(
#     weights_ptr,
#     v_ptr,
#     output_ptr,
#     seq_k,
#     head_dim,
#     stride_w0,
#     stride_w1,
#     stride_w2,
#     stride_v0,
#     stride_v1,
#     stride_v2,
#     stride_o0,
#     stride_o1,
#     stride_o2,
#     BLOCK_K: tl.constexpr,
#     BLOCK_D: tl.constexpr,
# ):
#     """Compute attention output: weights @ V."""
#     pid_bh = tl.program_id(0)
#     pid_q = tl.program_id(1)

#     offs_k = tl.arange(0, BLOCK_K)
#     offs_d = tl.arange(0, BLOCK_D)

#     w = tl.load(
#         weights_ptr
#         + pid_bh * stride_w0
#         + pid_q * stride_w1
#         + offs_k * stride_w2,
#         mask=offs_k < seq_k,
#         other=0.0,
#     )
#     v = tl.load(
#         v_ptr
#         + pid_bh * stride_v0
#         + offs_k[:, None] * stride_v1
#         + offs_d[None, :] * stride_v2,
#         mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim),
#         other=0.0,
#     )
#     out = tl.sum(v * w[:, None], axis=0)
#     tl.store(
#         output_ptr
#         + pid_bh * stride_o0
#         + pid_q * stride_o1
#         + offs_d * stride_o2,
#         out,
#         mask=offs_d < head_dim,
#     )


@triton.jit
def causal_mask_kernel(
    scores_ptr,
    seq_k,
    offset,
    stride_s0,
    stride_s1,
    stride_s2,
    BLOCK_K: tl.constexpr,
):
    """Apply causal mask to attention scores."""
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    mask = offs_k < seq_k
    scores = tl.load(
        scores_ptr
        + pid_bh * stride_s0
        + pid_q * stride_s1
        + offs_k * stride_s2,
        mask=mask,
        other=-1e9,
    )
    current_pos = pid_q + offset
    scores = tl.where(offs_k > current_pos, -1e9, scores)
    tl.store(
        scores_ptr
        + pid_bh * stride_s0
        + pid_q * stride_s1
        + offs_k * stride_s2,
        scores,
        mask=mask,
    )


# ============================================================================
# Layer Classes
# ============================================================================

def _is_power_of_two(x: int) -> bool:
    """Check if x is a power of two."""
    return x > 0 and (x & (x - 1)) == 0


class RMSNorm:
    """Root Mean Square Normalization using Triton with Torch fallback."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = torch.ones(hidden_size, dtype=torch.float32)
        self.use_triton = True  # Always use Triton — kernel handles non-power-of-2 via masking

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape

        if self.use_triton and x.is_cuda:
            batch_size = int(np.prod(x.shape[:-1]))
            # Pass native dtype (bf16) — kernel upcasts to fp32 internally
            # via .to(tl.float32). Avoids host-side bf16→fp32 cast on input.
            x_flat = x.reshape(batch_size, self.hidden_size).contiguous()
            output = torch.empty(
                (batch_size, self.hidden_size), dtype=torch.float32, device=x.device
            )

            if self.weight.device != x.device:
                self.weight = self.weight.to(x.device)

            block = next_power_of_two(self.hidden_size)
            rmsnorm_kernel[(batch_size,)](
                x_flat,
                self.weight,
                output,
                x_flat.stride(0),
                output.stride(0),
                self.hidden_size,
                self.eps,
                BLOCK_SIZE=block,
            )
            return output.reshape(original_shape).to(x.dtype)

        x_float = x.to(torch.float32)
        variance = torch.mean(x_float * x_float, dim=-1, keepdim=True)
        x_normed = x_float * torch.rsqrt(variance + self.eps)
        if self.weight.device != x.device:
            self.weight = self.weight.to(x.device)
        return (self.weight * x_normed).to(x.dtype)


class LayerNorm:
    """Layer Normalization using Triton with Torch fallback."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = torch.ones(hidden_size, dtype=torch.float32)
        self.bias = torch.zeros(hidden_size, dtype=torch.float32)
        self.use_triton = True  # Always use Triton — kernel handles non-power-of-2 via masking

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape

        if self.use_triton and x.is_cuda:
            batch_size = int(np.prod(x.shape[:-1]))
            # Pass native dtype (bf16) — kernel upcasts to fp32 internally.
            x_flat = x.reshape(batch_size, self.hidden_size).contiguous()
            output = torch.empty(
                (batch_size, self.hidden_size), dtype=torch.float32, device=x.device
            )

            if self.weight.device != x.device:
                self.weight = self.weight.to(x.device)
            if self.bias.device != x.device:
                self.bias = self.bias.to(x.device)

            block = next_power_of_two(self.hidden_size)
            layernorm_kernel[(batch_size,)](
                x_flat,
                self.weight,
                self.bias,
                output,
                x_flat.stride(0),
                output.stride(0),
                self.hidden_size,
                self.eps,
                BLOCK_SIZE=block,
            )
            return output.reshape(original_shape).to(x.dtype)

        x_float = x.to(torch.float32)
        mean = torch.mean(x_float, dim=-1, keepdim=True)
        variance = torch.var(x_float, dim=-1, keepdim=True, unbiased=False)
        x_normed = (x_float - mean) * torch.rsqrt(variance + self.eps)
        if self.weight.device != x.device:
            self.weight = self.weight.to(x.device)
        if self.bias.device != x.device:
            self.bias = self.bias.to(x.device)
        return (self.weight * x_normed + self.bias).to(x.dtype)


def gelu(x: torch.Tensor) -> torch.Tensor:
    """GELU activation using Triton."""
    original_shape = x.shape
    total = int(np.prod(x.shape))
    block = 1024

    x_flat = x.reshape(-1).contiguous()
    output = torch.empty_like(x_flat)
    grid = (triton.cdiv(total, block),)

    if x.is_cuda:
        gelu_kernel[grid](x_flat, output, total, BLOCK_SIZE=block)
        return output[:total].reshape(original_shape)

    return torch.nn.functional.gelu(x)


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU activation using Triton."""
    original_shape = x.shape
    total = int(np.prod(x.shape))
    block = 1024

    x_flat = x.reshape(-1).contiguous()
    output = torch.empty_like(x_flat)
    grid = (triton.cdiv(total, block),)

    if x.is_cuda:
        silu_kernel[grid](x_flat, output, total, BLOCK_SIZE=block)
        return output[:total].reshape(original_shape)

    return torch.nn.functional.silu(x)


def get_activation(name: str):
    """Get activation function by name."""
    activations = {"gelu": gelu, "silu": silu}
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name]


class Linear:
    """Linear layer with switchable backend (torch or Triton).

    Tile-size tuning (Optimization #1) — 12 configs benchmarked via tile_tune.py
    on NVIDIA H200 MIG 1g.18gb (Triton 3.6.0, PyTorch 2.10.0+cu128):

      Winner: TILE_M=32, TILE_N=32, TILE_K=32, num_warps=2, num_stages=2
        Sum latency 0.860 ms across all representative shapes (lowest of 12).
        Small tiles outperform larger ones on this MIG slice because the
        partition has few SMs — large tiles cause scheduling contention and
        exceed shared-memory limits.  See tile_tune_report.md for full table.
    """

    # ── Tile sizes (Optimization #1) ─────────────────────────────────────────
    # 12 configs benchmarked via tile_tune.py on NVIDIA H200 MIG 1g.18gb.
    # Winner: TILE_M=32, TILE_N=32, TILE_K=32, num_warps=2, num_stages=2
    #   audio-attn  (512×1280×1280): 0.156 ms
    #   audio-mlp   (512×1280×5120): 0.573 ms
    #   text-decode (1×3584×3584):   0.131 ms  → sum = 0.860 ms (lowest)
    # Small tiles win on MIG slice: fewer SMs means large tiles create
    # scheduling contention and increased shared-memory pressure.
    TILE_M = 32
    TILE_N = 32
    TILE_K = 32
    NUM_WARPS = 2
    NUM_STAGES = 2

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias

        self.weight = torch.zeros((out_features, in_features), dtype=torch.float32)
        self.bias_param = torch.zeros(out_features, dtype=torch.float32) if bias else None

        self._weight_t_padded = None
        self._K_padded = None
        self._N_padded = None

        # Deferred patch: inject generate_v8 onto GlmAsrModel once model.py is loaded
        _patch_model_generate()

    def _ensure_weight_prepared(self):
        """Cache transposed and padded weight for Triton kernel."""
        if self._weight_t_padded is None:
            K = self.in_features
            N = self.out_features
            self._K_padded = pad_to_multiple(K, self.TILE_K)
            self._N_padded = pad_to_multiple(N, self.TILE_N)

            weight_t = self.weight.t().contiguous()
            if self._K_padded > K or self._N_padded > N:
                weight_pad = torch.zeros(
                    (self._K_padded, self._N_padded),
                    dtype=self.weight.dtype,
                    device=weight_t.device,
                )
                weight_pad[:K, :N] = weight_t
                self._weight_t_padded = weight_pad
            else:
                self._weight_t_padded = weight_t

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        #if not x.is_cuda: - commented
        #    return self._forward_torch(x) - commented
        M = int(np.prod(x.shape[:-1]))
        #print(M, '=================================batchsize==================')
        # cuBLAS GEMV only for single-token decode (M <= 4).
        # A tiled GEMM kernel has ~1.5% utilisation at M=1 with TILE_M=32.
        #if M <= 4:#- commented
        #    print('using torch')
        #    return self._forward_torch(x) #- commented
        return self._forward_triton(x)

    def _forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        """Torch matmul backend."""
        original_shape = x.shape
        batch_dims = original_shape[:-1]

        M = int(np.prod(batch_dims))

        if self.weight.device != x.device:
            self.weight = self.weight.to(x.device)

        # Auto-convert weight to bf16 for 2× tensor core throughput on H200.
        # One-time lazy conversion; subsequent calls skip this branch.
        if self.weight.dtype == torch.float32 and x.is_cuda:
            self.weight = self.weight.to(torch.bfloat16)
            if self.has_bias and self.bias_param is not None:
                self.bias_param = self.bias_param.to(torch.bfloat16)
            self._weight_t_padded = None

        x_2d = x.reshape(M, self.in_features).to(self.weight.dtype)
        output = x_2d @ self.weight.t()

        if self.has_bias and self.bias_param is not None:
            if self.bias_param.device != x.device:
                self.bias_param = self.bias_param.to(x.device)
            output = output + self.bias_param

        return output.reshape(*batch_dims, self.out_features)

    def _forward_triton(self, x: torch.Tensor) -> torch.Tensor:
        """Triton matmul backend using linear_kernel_tf32."""
        original_shape = x.shape
        batch_dims = original_shape[:-1]

        M = int(np.prod(batch_dims))
        K = self.in_features
        N = self.out_features

        if self.weight.device != x.device:
            self.weight = self.weight.to(x.device)
            self._weight_t_padded = None

        # Lazy bf16 conversion: tl.dot on bf16 inputs uses bf16 tensor cores
        # with fp32 accumulation — same throughput as cuBLAS bf16 matmul.
        if self.weight.dtype == torch.float32:
            self.weight = self.weight.to(torch.bfloat16)
            if self.has_bias and self.bias_param is not None:
                self.bias_param = self.bias_param.to(torch.bfloat16)
            self._weight_t_padded = None

        self._ensure_weight_prepared()
        # Derive compute dtype from the actual padded weight tensor — guarantees
        # x_padded and _weight_t_padded have the same dtype for tl.dot.
        compute_dtype = self._weight_t_padded.dtype

        x_2d = x.reshape(M, K).to(compute_dtype).contiguous()

        M_padded = pad_to_multiple(M, self.TILE_M)

        if M_padded > M or self._K_padded > K:
            x_padded = torch.zeros(
                (M_padded, self._K_padded),
                dtype=compute_dtype,
                device=x.device,
            )
            x_padded[:M, :K] = x_2d
        else:
            x_padded = x_2d

        # Output stays fp32: kernel accumulator is tl.float32
        output = torch.zeros(
            (M_padded, self._N_padded), dtype=torch.float32, device=x.device
        )

        grid = (
            triton.cdiv(M_padded, self.TILE_M),
            triton.cdiv(self._N_padded, self.TILE_N),
        )
        linear_kernel_tf32[grid](
            x_padded,
            self._weight_t_padded,
            output,
            M_padded,
            self._N_padded,
            self._K_padded,
            x_padded.stride(0),
            x_padded.stride(1),
            self._weight_t_padded.stride(0),
            self._weight_t_padded.stride(1),
            output.stride(0),
            output.stride(1),
            BLOCK_M=self.TILE_M,
            BLOCK_N=self.TILE_N,
            BLOCK_K=self.TILE_K,
            num_warps=self.NUM_WARPS,
            num_stages=self.NUM_STAGES,
        )

        output = output[:M, :N]

        if self.has_bias and self.bias_param is not None:
            if self.bias_param.device != x.device:
                self.bias_param = self.bias_param.to(x.device)
            output = output + self.bias_param.to(torch.float32)

        # Return in weight dtype (bf16) to match torch backend behaviour.
        # (x.dtype may be float32 from upstream Conv1d; weight dtype is the
        # authoritative model precision after lazy bf16 conversion.)
        return output.reshape(*batch_dims, self.out_features).to(compute_dtype)


class Embedding:
    """Embedding layer using Triton."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.zeros((num_embeddings, embedding_dim), dtype=torch.float32)

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        original_shape = input_ids.shape
        batch_size = int(np.prod(original_shape))

        if self.weight.device != input_ids.device:
            self.weight = self.weight.to(input_ids.device)

        # Auto-convert embedding weight to bf16 for reduced HBM traffic.
        if self.weight.dtype == torch.float32 and input_ids.is_cuda:
            self.weight = self.weight.to(torch.bfloat16)

        if not input_ids.is_cuda:
            flat = input_ids.reshape(-1).to(torch.int64)
            output = self.weight.index_select(0, flat)
            return output.reshape(*original_shape, self.embedding_dim)

        indices_flat = input_ids.reshape(-1).to(torch.int32).contiguous()
        output = torch.empty(
            (batch_size, self.embedding_dim), dtype=self.weight.dtype, device=indices_flat.device
        )

        block = 1024
        grid = (batch_size, triton.cdiv(self.embedding_dim, block))
        embedding_kernel[grid](
            indices_flat,
            self.weight,
            output,
            self.embedding_dim,
            self.weight.stride(0),
            self.weight.stride(1),
            output.stride(0),
            BLOCK_SIZE=block,
        )

        return output.reshape(*original_shape, self.embedding_dim)


def softmax(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """Softmax using Triton kernel."""
    if axis != -1 and axis != len(x.shape) - 1:
        x = torch.movedim(x, axis, -1)

    original_shape = x.shape
    batch_size = int(np.prod(x.shape[:-1]))
    seq_len = x.shape[-1]

    x_flat = x.reshape(batch_size, seq_len).to(torch.float32).contiguous()
    output = torch.empty_like(x_flat)

    if x.is_cuda:
        print(f"Launching softmax kernel with batch_size={batch_size}, seq_len={seq_len}")
        block = next_power_of_two(seq_len)
        softmax_kernel[(batch_size,)]( #how is this even used when it is commented in the code above?
            x_flat,
            output,
            x_flat.stride(0),
            output.stride(0),
            seq_len,
            BLOCK_SIZE=block,
        )
        result = output.reshape(original_shape)
    else:
        result = torch.softmax(x, dim=-1)

    if axis != -1 and axis != len(original_shape) - 1:
        result = torch.movedim(result, -1, axis)

    return result


class MLP:
    """MLP with SwiGLU gating using Triton.

    Uses swiglu_fused_kernel (Optimization #2): fuses gate + up projections
    and SiLU activation into a single Triton kernel pass.
    Tile sizes follow Linear class (Optimization #1): 64/64/32 default.
    """

    FUSED = True
    # Tuned via kernel_tune.py on NVIDIA H200 MIG 1g.18gb.
    # Best config: M32-N64(32,64,32) — sum latency 4.743 ms across all shapes.
    # num_warps=4, num_stages=2 from MatmulConfig("M32-N64", 32, 64, 32, 4, 2).
    TILE_M, TILE_N, TILE_K = 32, 64, 32
    NUM_WARPS, NUM_STAGES = 4, 2

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "silu",
        bias: bool = False,
        use_gating: bool = True,
    ):
        self.use_gating = use_gating
        self.act_fn = get_activation(activation)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.bias_enabled = bias

        if use_gating:
            self.gate_proj = Linear(hidden_size, intermediate_size, bias=bias)
            self.up_proj = Linear(hidden_size, intermediate_size, bias=bias)
        else:
            self.up_proj = Linear(hidden_size, intermediate_size, bias=bias)

        self.down_proj = Linear(intermediate_size, hidden_size, bias=bias)

        self._gate_weight_t = None
        self._up_weight_t = None

    def _prepare_fused_weights(self):
        """Prepare pre-transposed weights for fused kernel."""
        if self._gate_weight_t is None and self.use_gating:
            if self.gate_proj.weight.device != self.up_proj.weight.device:
                self.up_proj.weight = self.up_proj.weight.to(self.gate_proj.weight.device)
            self._gate_weight_t = self.gate_proj.weight.t().contiguous()
            self._up_weight_t = self.up_proj.weight.t().contiguous()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_gating and MLP.FUSED and x.is_cuda:
            return self._forward_fused(x)
        return self._forward_standard(x)

    def _forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        """Standard (unfused) forward pass."""
        if self.use_gating:
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return self.down_proj(self.act_fn(self.up_proj(x)))

    def _forward_fused(self, x: torch.Tensor) -> torch.Tensor:
        """Fused SwiGLU forward pass."""
        M = int(np.prod(x.shape[:-1]))
        # Removing Fallback to Torch
        # Uncommenting reduces time by ~2% (~6-10ms / 458ms→466ms)
        # if M <= 16:
        #     return self._forward_standard(x)

        if self.gate_proj.weight.device != x.device:
            self.gate_proj.weight = self.gate_proj.weight.to(x.device)
            self._gate_weight_t = None
        if self.up_proj.weight.device != x.device:
            self.up_proj.weight = self.up_proj.weight.to(x.device)
            self._up_weight_t = None
        self._prepare_fused_weights()

        orig_shape = x.shape
        # Auto-convert fused weights to bf16 for 2× tensor core throughput.
        if self._gate_weight_t is not None and self._gate_weight_t.dtype == torch.float32 and x.is_cuda:
            self._gate_weight_t = self._gate_weight_t.to(torch.bfloat16)
            self._up_weight_t = self._up_weight_t.to(torch.bfloat16)
        compute_dtype = self._gate_weight_t.dtype if self._gate_weight_t is not None else torch.float32
        x_2d = x.reshape(-1, self.hidden_size).to(compute_dtype).contiguous()
        M = x_2d.shape[0]
        K = self.hidden_size
        N = self.intermediate_size

        M_pad = pad_to_multiple(M, self.TILE_M)
        K_pad = pad_to_multiple(K, self.TILE_K)
        N_pad = pad_to_multiple(N, self.TILE_N)

        if M != M_pad or K != K_pad:
            x_padded = torch.zeros(
                (M_pad, K_pad), dtype=compute_dtype, device=x.device
            )
            x_padded[:M, :K] = x_2d
        else:
            x_padded = x_2d

        if K != K_pad or N != N_pad:
            gate_w_padded = torch.zeros(
                (K_pad, N_pad), dtype=compute_dtype, device=x.device
            )
            gate_w_padded[:K, :N] = self._gate_weight_t
            up_w_padded = torch.zeros(
                (K_pad, N_pad), dtype=compute_dtype, device=x.device
            )
            up_w_padded[:K, :N] = self._up_weight_t
        else:
            gate_w_padded = self._gate_weight_t
            up_w_padded = self._up_weight_t

        intermediate = torch.zeros(
            (M_pad, N_pad), dtype=torch.float32, device=x.device
        )

        grid = (
            triton.cdiv(M_pad, self.TILE_M),
            triton.cdiv(N_pad, self.TILE_N),
        )
        swiglu_fused_kernel[grid](
            x_padded,
            gate_w_padded,
            up_w_padded,
            intermediate,
            M_pad,
            N_pad,
            K_pad,
            x_padded.stride(0),
            x_padded.stride(1),
            gate_w_padded.stride(0),
            gate_w_padded.stride(1),
            up_w_padded.stride(0),
            up_w_padded.stride(1),
            intermediate.stride(0),
            intermediate.stride(1),
            BLOCK_M=self.TILE_M,
            BLOCK_N=self.TILE_N,
            BLOCK_K=self.TILE_K,
            num_warps=self.NUM_WARPS,
            num_stages=self.NUM_STAGES,
        )

        if M != M_pad or N != N_pad:
            intermediate = intermediate[:M, :N]

        intermediate = intermediate.reshape(*orig_shape[:-1], self.intermediate_size)
        return self.down_proj(intermediate)


class EncoderMLP:
    """Encoder MLP (no gating) using Triton.

    Uses linear_gelu_kernel (Optimization #2): fuses Linear + GELU
    into a single Triton kernel pass (avoids intermediate HBM round-trip).
    Tile sizes follow Linear class (Optimization #1): 64/64/32 default.
    """

    FUSED = True
    # Tuned via kernel_tune.py on NVIDIA H200 MIG 1g.18gb.
    # Best config: Tiny(32,32,32) — sum latency 1.553 ms across all shapes.
    # num_warps=2, num_stages=2 from MatmulConfig("Tiny", 32, 32, 32, 2, 2).
    TILE_M, TILE_N, TILE_K = 32, 32, 32
    NUM_WARPS, NUM_STAGES = 2, 2

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "gelu",
        bias: bool = True,
    ):
        self.fc1 = Linear(hidden_size, intermediate_size, bias=bias)
        self.fc2 = Linear(intermediate_size, hidden_size, bias=bias)
        self.act_fn = get_activation(activation)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.bias_enabled = bias
        self.activation = activation

        self._fc1_weight_t = None

    def _prepare_fused_weights(self):
        """Prepare pre-transposed weights for fused kernel."""
        if self._fc1_weight_t is None:
            self._fc1_weight_t = self.fc1.weight.t().contiguous()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if EncoderMLP.FUSED and self.activation == "gelu" and x.is_cuda:
            return self._forward_fused(x)
        return self._forward_standard(x)

    def _forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        """Standard (unfused) forward pass."""
        return self.fc2(self.act_fn(self.fc1(x)))

    def _forward_fused(self, x: torch.Tensor) -> torch.Tensor:
        """Fused Linear+GELU forward pass."""
        if self.fc1.weight.device != x.device:
            self.fc1.weight = self.fc1.weight.to(x.device)
            self._fc1_weight_t = None
        self._prepare_fused_weights()

        orig_shape = x.shape
        # Auto-convert fused weights to bf16 for 2× tensor core throughput.
        if self._fc1_weight_t is not None and self._fc1_weight_t.dtype == torch.float32 and x.is_cuda:
            self._fc1_weight_t = self._fc1_weight_t.to(torch.bfloat16)
        compute_dtype = self._fc1_weight_t.dtype if self._fc1_weight_t is not None else torch.float32
        x_2d = x.reshape(-1, self.hidden_size).to(compute_dtype).contiguous()
        M = x_2d.shape[0]
        K = self.hidden_size
        N = self.intermediate_size

        M_pad = pad_to_multiple(M, self.TILE_M)
        K_pad = pad_to_multiple(K, self.TILE_K)
        N_pad = pad_to_multiple(N, self.TILE_N)

        if M != M_pad or K != K_pad:
            x_padded = torch.zeros(
                (M_pad, K_pad), dtype=compute_dtype, device=x.device
            )
            x_padded[:M, :K] = x_2d
        else:
            x_padded = x_2d

        if K != K_pad or N != N_pad:
            fc1_w_padded = torch.zeros(
                (K_pad, N_pad), dtype=compute_dtype, device=x.device
            )
            fc1_w_padded[:K, :N] = self._fc1_weight_t
        else:
            fc1_w_padded = self._fc1_weight_t

        intermediate = torch.zeros(
            (M_pad, N_pad), dtype=torch.float32, device=x.device
        )

        grid = (
            triton.cdiv(M_pad, self.TILE_M),
            triton.cdiv(N_pad, self.TILE_N),
        )
        linear_gelu_kernel[grid](
            x_padded,
            fc1_w_padded,
            intermediate,
            M_pad,
            N_pad,
            K_pad,
            x_padded.stride(0),
            x_padded.stride(1),
            fc1_w_padded.stride(0),
            fc1_w_padded.stride(1),
            intermediate.stride(0),
            intermediate.stride(1),
            BLOCK_M=self.TILE_M,
            BLOCK_N=self.TILE_N,
            BLOCK_K=self.TILE_K,
            num_warps=self.NUM_WARPS,
            num_stages=self.NUM_STAGES,
        )

        if M != M_pad or N != N_pad:
            intermediate = intermediate[:M, :N]

        if self.bias_enabled and self.fc1.bias_param is not None:
            if self.fc1.bias_param.device != x.device:
                self.fc1.bias_param = self.fc1.bias_param.to(x.device)
            intermediate = intermediate + self.fc1.bias_param

        intermediate = intermediate.reshape(*orig_shape[:-1], self.intermediate_size)
        return self.fc2(intermediate)


# ============================================================================
# Optimized Generate v8 (monkey-patched onto GlmAsrModel)
# ============================================================================
# Same logic as model.py generate() but with:
#   1. torch.inference_mode() — disables autograd dispatch overhead
#   2. Fast greedy path for top_k <= 1 — argmax instead of argsort(151K vocab)
#   3. torch.topk for general top_k — O(n) vs O(n log n) argsort
#   4. Pre-allocated generated token list — avoids 12× torch.cat on growing tensor

_MODEL_PATCHED = False


@torch.inference_mode()
def _generate_v8_impl(self, input_features, input_ids=None,
                       input_features_mask=None, attention_mask=None,
                       max_new_tokens=256, temperature=1.0, top_k=50,
                       audio_pad_token_id=59260):
    """Optimized autoregressive generation — inference_mode + fast sampling."""
    # ── Encode audio (identical to generate()) ───────────────────────────
    audio_embeds = self.encode_audio(input_features, input_features_mask)

    if input_ids is not None:
        batch_size = input_ids.shape[0]
        if audio_embeds.ndim == 3:
            audio_embeds = audio_embeds[0]
        text_embeds = self.text_decoder.embed_tokens(input_ids)
        audio_mask = (input_ids == audio_pad_token_id)
        audio_positions = torch.where(audio_mask[0])[0]

        if len(audio_positions) > 0:
            first_pad_pos = int(audio_positions[0].item())
            last_pad_pos = int(audio_positions[-1].item())
            before_audio = text_embeds[0, :first_pad_pos, :]
            after_audio = text_embeds[0, last_pad_pos + 1:, :]
            inputs_embeds = torch.cat([
                before_audio[None, :, :],
                audio_embeds[None, :, :],
                after_audio[None, :, :],
            ], dim=1)
        else:
            inputs_embeds = text_embeds
        generated = input_ids.clone()
    else:
        batch_size = audio_embeds.shape[0] if audio_embeds.ndim == 3 else 1
        if audio_embeds.ndim == 2:
            audio_embeds = audio_embeds[None, :, :]
        inputs_embeds = audio_embeds
        generated = torch.full(
            (batch_size, 1), self.config.bos_token_id,
            dtype=torch.int64, device=inputs_embeds.device,
        )

    # ── EOS setup ────────────────────────────────────────────────────────
    finished = torch.zeros(batch_size, dtype=torch.bool, device=generated.device)
    eos_token_ids = self.config.eos_token_id
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    eos_tensor = torch.tensor(eos_token_ids, dtype=torch.int64, device=generated.device)

    # ── Pre-compute sampling mode ────────────────────────────────────────
    greedy = (top_k <= 1) or (temperature == 0)
    vocab_size = None  # lazily determined

    # ── Collect tokens in list, join at end (avoids 12× torch.cat) ───────
    generated_tokens = []

    # ── Autoregressive generation (no KV cache — full decode each step) ──
    for _ in range(max_new_tokens):
        logits = self.decode(inputs_embeds=inputs_embeds)
        next_token_logits = logits[:, -1, :]

        if greedy:
            # Fast path: pure argmax — skips argsort/gather/softmax/cumsum
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            next_token_logits = next_token_logits / temperature
            # Use torch.topk instead of argsort — O(n) vs O(n log n)
            if vocab_size is None:
                vocab_size = next_token_logits.shape[-1]
            k = min(top_k, vocab_size)
            top_k_logits, top_k_indices = torch.topk(next_token_logits, k, dim=-1)

            # Softmax over top-k
            top_k_logits = top_k_logits - top_k_logits.max(dim=-1, keepdim=True).values
            exp_logits = torch.exp(top_k_logits)
            probs = exp_logits / exp_logits.sum(dim=-1, keepdim=True)

            # Categorical sampling
            cumprobs = torch.cumsum(probs, dim=-1)
            samples = torch.rand((batch_size, 1), device=next_token_logits.device)
            next_token_idx = torch.argmax((cumprobs >= samples).to(torch.float32), dim=-1)
            next_token = torch.gather(
                top_k_indices, dim=-1, index=next_token_idx[:, None],
            )

        generated_tokens.append(next_token)

        # EOS check
        next_token_flat = next_token.flatten()
        is_eos = torch.any(
            next_token_flat[:, None] == eos_tensor[None, :], dim=1
        )
        finished = finished | is_eos
        if torch.all(finished):
            break

        # Update inputs_embeds with new token (full sequence reprocessed)
        new_embeds = self.text_decoder.embed_tokens(next_token)
        inputs_embeds = torch.cat([inputs_embeds, new_embeds], dim=1)

    # ── Join all generated tokens at once ────────────────────────────────
    if generated_tokens:
        generated = torch.cat([generated] + generated_tokens, dim=1)

    return generated


def _patch_model_generate():
    """Deferred monkey-patch: inject generate_v8 onto GlmAsrModel."""
    global _MODEL_PATCHED
    if _MODEL_PATCHED:
        return
    try:
        import sys
        model_mod = sys.modules.get('model')
        if model_mod is None:
            return
        GlmAsrModel = getattr(model_mod, 'GlmAsrModel', None)
        if GlmAsrModel is None:
            return
        if not hasattr(GlmAsrModel, 'generate_v8'):
            GlmAsrModel.generate_v8 = _generate_v8_impl
        _MODEL_PATCHED = True
    except Exception:
        pass


if __name__ == "__main__":
    print("Testing Triton Layers...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n=== RMSNorm ===")
    norm = RMSNorm(256)
    x = torch.randn(2, 16, 256, device=device, dtype=torch.float32)
    y = norm(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")

    print("\n=== LayerNorm ===")
    ln = LayerNorm(256)
    y = ln(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")

    print("\n=== GELU ===")
    y = gelu(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")

    print("\n=== SiLU ===")
    y = silu(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")

    print("\n=== Linear ===")
    linear = Linear(256, 512)
    y = linear(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")

    print("\n=== Embedding ===")
    emb = Embedding(1000, 256)
    ids = torch.randint(0, 1000, (2, 16), device=device, dtype=torch.int32)
    y = emb(ids)
    print(f"Input: {ids.shape} -> Output: {y.shape}")

    print("\n=== Softmax ===")
    x_sm = torch.randn(2, 4, 16, 16, device=device, dtype=torch.float32)
    y = softmax(x_sm, axis=-1)
    print(f"Input: {x_sm.shape} -> Output: {y.shape}")
    print(f"Sum along last axis: {float(y[0, 0, 0].sum()):.6f} (should be 1.0)")

    print("\n=== MLP ===")
    mlp = MLP(256, 512, activation="silu", use_gating=True)
    y = mlp(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")

    print("\nAll Triton layers working!")
