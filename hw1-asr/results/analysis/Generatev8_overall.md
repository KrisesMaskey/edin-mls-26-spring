# HW1-ASR: Generation Optimization Report (`generate_v8`)

---

## 1. Objective

This report evaluates the impact of an optimized generation pipeline:

* Baseline: `generate`
* Optimized: `_generate_v8_impl`

The goal is to understand:

* End-to-end latency improvements
* Operator-level bottlenecks
* Design-level optimizations beyond kernel tuning

---

## 2. Summary of Results

### 2.1 Without `generate_v8` (Baseline)

```
Time: 521.1ms (+/- 20.9ms)
Tokens: 13
Speed: 40.08ms/token

Accuracy: 100.0%
Status: PASS
```

---

### 2.2 With `generate_v8` (Optimized)

```
Time: 291.0ms (+/- 0.6ms)
Tokens: 13
Speed: 22.38ms/token

Accuracy: 100.0%
Status: PASS
```

---

## 3. Performance Improvement

| Metric     | Baseline | Optimized | Improvement       |
| ---------- | -------- | --------- | ----------------- |
| Total Time | 521.1 ms | 291.0 ms  | **~1.8× faster**  |
| Per Token  | 40.08 ms | 22.38 ms  | **~1.79× faster** |
| Variance   | High     | Low       | **More stable**   |

---

## 4. Detailed Profiling Comparison

### 4.1 Without `generate_v8`

#### Total Estimated Runtime (50 tokens)

```
TOTAL: 17047.28 ms
```

| Component          | Time        | %     |
| ------------------ | ----------- | ----- |
| Audio Encoder      | 2799.88 ms  | 16.4% |
| Projector          | 23.74 ms    | 0.1%  |
| Decoder Prefill    | 751.28 ms   | 4.4%  |
| Decoder (50 steps) | 13472.39 ms | 79.0% |

---

### 4.2 With `generate_v8`

#### Total Estimated Runtime (50 tokens)

```
TOTAL: 5025.42 ms
```

| Component          | Time       | %     |
| ------------------ | ---------- | ----- |
| Audio Encoder      | 2759.26 ms | 54.9% |
| Projector          | 23.09 ms   | 0.5%  |
| Decoder Prefill    | 702.58 ms  | 14.0% |
| Decoder (50 steps) | 1540.49 ms | 30.7% |

---

## 5. Key Observations

### 5.1 Massive Decode Optimization 🚀

| Configuration         | Decode Time |
| --------------------- | ----------- |
| Without `generate_v8` | 13472 ms    |
| With `generate_v8`    | 1540 ms     |

> ✅ **~8.7× speedup in decoding**

---

### 5.2 Bottleneck Shift

* **Before:**

  * Decoder dominates (**79%**)

* **After:**

  * Audio Encoder dominates (**54.9%**)

> 🔄 Optimization shifts bottleneck from **decoder → encoder**

---

### 5.3 Reduced Variance

* Baseline: ±20.9 ms
* Optimized: ±0.6 ms

> ✅ More stable runtime due to fewer dynamic operations

---

### 5.4 Attention & GEMM Insights

* Torch matmul attention: ~0.6–0.9 ms
* Standard einsum: ~180–220 ms

> Confirms that **implementation choice dominates performance**

---

## 6. `generate_v8` Implementation

### 6.1 Core Function

```python
@torch.inference_mode()
def _generate_v8_impl(self, input_features, input_ids=None,
                       input_features_mask=None, attention_mask=None,
                       max_new_tokens=256, temperature=1.0, top_k=50,
                       audio_pad_token_id=59260):

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

    finished = torch.zeros(batch_size, dtype=torch.bool, device=generated.device)
    eos_token_ids = self.config.eos_token_id
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    eos_tensor = torch.tensor(eos_token_ids, dtype=torch.int64, device=generated.device)

    greedy = (top_k <= 1) or (temperature == 0)
    vocab_size = None
    generated_tokens = []

    for _ in range(max_new_tokens):
        logits = self.decode(inputs_embeds=inputs_embeds)
        next_token_logits = logits[:, -1, :]

        if greedy:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            next_token_logits = next_token_logits / temperature
            if vocab_size is None:
                vocab_size = next_token_logits.shape[-1]
            k = min(top_k, vocab_size)
            top_k_logits, top_k_indices = torch.topk(next_token_logits, k, dim=-1)

            top_k_logits = top_k_logits - top_k_logits.max(dim=-1, keepdim=True).values
            exp_logits = torch.exp(top_k_logits)
            probs = exp_logits / exp_logits.sum(dim=-1, keepdim=True)

            cumprobs = torch.cumsum(probs, dim=-1)
            samples = torch.rand((batch_size, 1), device=next_token_logits.device)
            next_token_idx = torch.argmax((cumprobs >= samples).to(torch.float32), dim=-1)
            next_token = torch.gather(top_k_indices, dim=-1, index=next_token_idx[:, None])

        generated_tokens.append(next_token)

        next_token_flat = next_token.flatten()
        is_eos = torch.any(next_token_flat[:, None] == eos_tensor[None, :], dim=1)
        finished = finished | is_eos
        if torch.all(finished):
            break

        new_embeds = self.text_decoder.embed_tokens(next_token)
        inputs_embeds = torch.cat([inputs_embeds, new_embeds], dim=1)

    if generated_tokens:
        generated = torch.cat([generated] + generated_tokens, dim=1)

    return generated
```

---

## 7. Key Optimizations in `generate_v8`

### 7.1 Inference Mode

* Disables autograd → faster execution

### 7.2 Greedy Fast Path

* Uses `argmax` → avoids softmax + sampling

### 7.3 Top-K Optimization

* Uses `torch.topk` instead of `argsort`

### 7.4 Reduced Concatenation Overhead

* Collect tokens in list → single `torch.cat`

### 7.5 Early Stopping

* Stops when EOS reached

---

## 8. Limitations

* No KV cache → recomputes full sequence
* Still sequential decoding
* Further gains possible with caching

---

## 9. Final Conclusion

> `generate_v8` provides **major system-level speedups (~1.8× end-to-end, ~8× decode)** without modifying model architecture.

### Key Insight:

> **Optimizing execution flow can be as impactful as kernel optimization.**

---

## 10. Future Work

* Implement KV caching
* Optimize audio encoder (new bottleneck)
* Fuse MLP kernels

---
