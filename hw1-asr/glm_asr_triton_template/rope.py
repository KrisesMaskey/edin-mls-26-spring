"""
Triton Rotary Position Embeddings (RoPE)
End-to-end implementation using Triton kernels


*** STUDENT ASSIGNMENT ***
Fill in the TODO sections to implement RoPE using Triton kernels
"""


from typing import Optional, Tuple


import torch
import triton
import triton.language as tl




def get_stream():
    """Get current CUDA stream pointer."""
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None




# ============================================================================
# Triton Kernels for RoPE
# ============================================================================


# run the below autotune to get best block size


# def print_all_configs_bench(configs, named_args, **kwargs):
#     """Called before autotuning — we'll print results in post_hook instead."""
#     pass


# results_log = []  # global to collect results across hook calls


# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK': 16},  num_warps=1),
#         triton.Config({'BLOCK': 32},  num_warps=2),
#         triton.Config({'BLOCK': 64},  num_warps=4),
#         triton.Config({'BLOCK': 128}, num_warps=4),
#         triton.Config({'BLOCK': 256}, num_warps=8),
#     ],
#     key=['half_dim'],
#     prune_configs_by={
#         'early_config_prune': lambda configs, named_args, **kwargs: [
#             c for c in configs
#             if c.kwargs['BLOCK'] >= triton.next_power_of_2(named_args['half_dim'])
#         ]
#     }
# )




# @triton.heuristics({
#     'BLOCK': lambda args: 64,  # winner for all your half_dim cases
# })


@triton.jit
def compute_freqs_kernel(
    positions_ptr,
    inv_freq_ptr,
    cos_ptr,
    sin_ptr,
    seq_len,
    half_dim,
    stride_pos,
    stride_inv,
    stride_cos0,
    stride_cos1,
    stride_sin0,
    stride_sin1,
    BLOCK: tl.constexpr,
):
    """Compute cos and sin for rotary embeddings."""
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < half_dim


    pos = tl.load(positions_ptr + pid * stride_pos)
    inv = tl.load(inv_freq_ptr + offs * stride_inv, mask=mask, other=0.0)
    freqs = pos * inv


    cos_half = tl.cos(freqs)
    sin_half = tl.sin(freqs)


    tl.store(cos_ptr + pid * stride_cos0 + offs * stride_cos1, cos_half, mask=mask)
    tl.store(
        cos_ptr + pid * stride_cos0 + (offs + half_dim) * stride_cos1,
        cos_half,
        mask=mask,
    )
    tl.store(sin_ptr + pid * stride_sin0 + offs * stride_sin1, sin_half, mask=mask)
    tl.store(
        sin_ptr + pid * stride_sin0 + (offs + half_dim) * stride_sin1,
        sin_half,
        mask=mask,




     )
# @triton.jit
# def compute_freqs_kernel(
#     positions_ptr,
#     inv_freq_ptr,
#     cos_ptr,
#     sin_ptr,
#     seq_len,
#     half_dim,
#     stride_pos,
#     stride_inv,
#     stride_cos0,
#     stride_cos1,
#     stride_sin0,
#     stride_sin1,
#     BLOCK: tl.constexpr,
# ):
#     """
#     Compute cos and sin for rotary embeddings.


#     *** TODO: Implement this kernel ***


#     Grid: (seq_len,)
#     """
#     pid = tl.program_id(0)
#     offs_full = tl.arange(0, BLOCK*2)


#     mask_full = offs_full < (half_dim*2)


#     #tl.static_print("BLOCK =", BLOCK)  verify the right block


#     offs_mod = offs_full % half_dim
#     #offs_mod = tl.where(offs_full < half_dim, offs_full, offs_full - half_dim)
   


#     pos = tl.load(positions_ptr + pid * stride_pos)
#     inv = tl.load(inv_freq_ptr + offs_mod * stride_inv, mask=mask_full, other=0.0)
#     freqs = pos * inv


#     cos_vals = tl.cos(freqs)
#     sin_vals = tl.sin(freqs)


#     tl.store(cos_ptr + pid * stride_cos0 + offs_full * stride_cos1, cos_vals, mask=mask_full)
#     tl.store(sin_ptr + pid * stride_sin0 + offs_full * stride_sin1, sin_vals, mask=mask_full)


# ============================================================================
# RoPE Classes
# ============================================================================


class RotaryEmbedding:
    """Rotary Position Embedding using Triton."""


    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
        base: float = 10000.0,
        partial_rotary_factor: float = 1.0,
    ):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.partial_rotary_factor = partial_rotary_factor


        self.rotary_dim = int(dim * partial_rotary_factor)
        self.rotary_dim = self.rotary_dim - (self.rotary_dim % 2)


       
        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )


   
        self.inv_freq = inv_freq


        self._update_cache(max_position_embeddings)


    def _update_cache(self, seq_len: int, device: Optional[torch.device] = None):
        """Pre-compute cos and sin using Triton kernel."""
        self.max_seq_len_cached = seq_len
        half_dim = self.rotary_dim // 2
        if device is None:
            device = self.inv_freq.device


        if self.inv_freq.device != device:
            self.inv_freq = self.inv_freq.to(device, non_blocking=True)


        #converting  to float16 for benchmarking the kernel, you can remove this for final implementation
        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        cos_cache = torch.empty((seq_len, self.rotary_dim), dtype=torch.float32, device=device)
        sin_cache = torch.empty((seq_len, self.rotary_dim), dtype=torch.float32, device=device)


       
        # if device.type == "cuda":
        #     if self.inv_freq.device != device:
        #         self.inv_freq = self.inv_freq.to(device)
        if device.type == "cuda":
            #block = triton.next_power_of_2(half_dim)


            block = triton.next_power_of_2(half_dim)# remove for autotune
            num_warps = 1 if block <= 32 else 2
            compute_freqs_kernel[(seq_len,)](
                positions,
                self.inv_freq,
                cos_cache,
                sin_cache,
                seq_len,
                half_dim,
                positions.stride(0),
                self.inv_freq.stride(0),
                cos_cache.stride(0),
                cos_cache.stride(1),
                sin_cache.stride(0),
                sin_cache.stride(1),
                BLOCK=block, #remove for autotune
                num_warps=num_warps, # <-- Prevents thread idling
                num_stages=2,
            )
        else:
            if self.inv_freq.device != device:
                self.inv_freq = self.inv_freq.to(device)
            freqs = positions[:, None] * self.inv_freq[None, :]
            cos_half = torch.cos(freqs)
            sin_half = torch.sin(freqs)
            cos_cache[:, :half_dim] = cos_half
            cos_cache[:, half_dim : half_dim * 2] = cos_half
            sin_cache[:, :half_dim] = sin_half
            sin_cache[:, half_dim : half_dim * 2] = sin_half


        # Add this after the kernel call in _update_cache this is for benchmarking the winnign config
   


        self.cos_cached = cos_cache
        self.sin_cached = sin_cache


        #run below to get best resulting block


        # if device.type == "cuda":
        #     all_configs = [
        #         triton.Config({'BLOCK': 16},  num_warps=1),
        #         triton.Config({'BLOCK': 32},  num_warps=2),
        #         triton.Config({'BLOCK': 64},  num_warps=4),
        #         triton.Config({'BLOCK': 128}, num_warps=4),
        #         triton.Config({'BLOCK': 256}, num_warps=8),
        #     ]


        #     # Only benchmark valid configs for this half_dim
        #     valid_configs = [
        #         c for c in all_configs
        #         if c.kwargs['BLOCK'] >= triton.next_power_of_2(half_dim)
        #     ]


        #     print(f"\n  Benchmarking all configs for half_dim={half_dim}, seq_len={seq_len}")
        #     print(f"  {'BLOCK':<8} {'num_warps':<12} {'latency (ms)':<16} {'GB/s':<10} {'winner?'}")
        #     print(f"  {'-'*58}")


        #     bytes_total = (seq_len * 4 + half_dim * 4) + (seq_len * self.rotary_dim * 4 * 2)


        #     best_block = None
        #     best_warps = None
        #     if compute_freqs_kernel.cache:
        #         for k, v in compute_freqs_kernel.cache.items():
        #             if k[0] == half_dim:
        #                 best_block = v.kwargs['BLOCK']
        #                 best_warps = v.num_warps


        #     for cfg in valid_configs:
        #         block     = cfg.kwargs['BLOCK']
        #         num_warps = cfg.num_warps


        #         try:
        #             ms = triton.testing.do_bench(
        #                 lambda b=block, nw=num_warps: compute_freqs_kernel.fn[(seq_len,)](
        #                     positions,
        #                     self.inv_freq,
        #                     cos_cache,
        #                     sin_cache,
        #                     seq_len,
        #                     half_dim,
        #                     positions.stride(0),
        #                     self.inv_freq.stride(0),
        #                     cos_cache.stride(0),
        #                     cos_cache.stride(1),
        #                     sin_cache.stride(0),
        #                     sin_cache.stride(1),
        #                     BLOCK=b,
        #                     num_warps=nw,
        #                 ),
        #                 warmup=25,
        #                 rep=100,
        #             )
        #             gb_s    = bytes_total / (ms * 1e-3) / 1e9
        #             is_best = (block == best_block and num_warps == best_warps)
        #             marker  = "<-- best" if is_best else ""
        #             print(f"  {block:<8} {num_warps:<12} {ms:<16.4f} {gb_s:<10.2f} {marker}")


        #         except Exception as e:
        #             print(f"  {block:<8} {num_warps:<12} {'ERROR':<16} {'N/A':<10} {e}")


        #     print()


    def __call__(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cos and sin for given positions."""
        seq_len = x.shape[-2]


        if seq_len > self.max_seq_len_cached:
            self._update_cache(seq_len, device=x.device)
        elif self.cos_cached.device != x.device:
            self._update_cache(self.max_seq_len_cached, device=x.device)


        if position_ids is not None:
            cos = self.cos_cached[position_ids].to(x.dtype)
            sin = self.sin_cached[position_ids].to(x.dtype)
            if cos.ndim == 3 and cos.shape[0] == 1:
                cos = cos[0]
                sin = sin[0]
        else:
            cos = self.cos_cached[:seq_len].to(x.dtype)
            sin = self.sin_cached[:seq_len].to(x.dtype)


        return cos, sin




def next_power_of_two(x: int) -> int:
    """Return the smallest power of two >= x."""
    return 1 << (x - 1).bit_length() if x > 0 else 1




MAX_ROPE_DIM = 256




def _apply_rope_single(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    half_dim: int,
    head_dim: int,
) -> torch.Tensor:
    """Apply RoPE to a single tensor (Q or K) using Torch."""
    batch, num_heads, seq_len, _ = x.shape


    cos = cos[:seq_len]
    sin = sin[:seq_len]


    x1 = x[..., :half_dim]
    x2 = x[..., half_dim : half_dim * 2]


    cos_expanded = cos[None, None, :, :]
    sin_expanded = sin[None, None, :, :]


    x1_rot = x1 * cos_expanded - x2 * sin_expanded
    x2_rot = x2 * cos_expanded + x1 * sin_expanded


    if head_dim > half_dim * 2:
        x_pass = x[..., half_dim * 2 :]
        return torch.cat([x1_rot, x2_rot, x_pass], dim=-1)
    return torch.cat([x1_rot, x2_rot], dim=-1)




def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings.
    """
    batch, num_q_heads, seq_len, head_dim = q.shape
    _, num_kv_heads, _, _ = k.shape


    if rotary_dim is None:
        rotary_dim = head_dim


    half_dim = rotary_dim // 2


    if cos.shape[1] > half_dim:
        cos = cos[:, :half_dim]
        sin = sin[:, :half_dim]


    cos = cos.to(torch.float32).contiguous()
    sin = sin.to(torch.float32).contiguous()


    q_out = _apply_rope_single(q, cos, sin, half_dim, head_dim)
    k_out = _apply_rope_single(k, cos, sin, half_dim, head_dim)


    return q_out.to(q.dtype), k_out.to(k.dtype)




def apply_partial_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to partial dimensions."""
    return apply_rotary_pos_emb(q, k, cos, sin, rotary_dim)




if __name__ == "__main__":
    print("Testing Triton RoPE...")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    num_heads = 4
    seq_len = 16
    head_dim = 64


    rope = RotaryEmbedding(dim=head_dim, max_position_embeddings=1024)


    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)


    cos, sin = rope(q)
    print(f"Cos shape: {cos.shape}")
    print(f"Sin shape: {sin.shape}")


    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    print(f"Q rotated shape: {q_rot.shape}")
    print(f"K rotated shape: {k_rot.shape}")


    print("\nTesting partial RoPE (50%):")
    rope_partial = RotaryEmbedding(dim=head_dim, partial_rotary_factor=0.5)
    cos_p, sin_p = rope_partial(q)
    q_rot_p, k_rot_p = apply_partial_rotary_pos_emb(q, k, cos_p, sin_p, head_dim // 2)
    print(f"Q rotated (partial) shape: {q_rot_p.shape}")


    print("\nTriton RoPE working!")