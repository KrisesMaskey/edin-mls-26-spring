# README-FIRST — GLM-ASR Submission Guide

## Track

We implemented the **Triton track** (`glm_asr_triton_template`).

---

## Implementation Files

The three core files we implemented are:

| File | Description |
|------|-------------|
| `glm_asr_triton_template/attention.py` | FlashAttention-style fused attention kernel with online softmax and causal masking |
| `glm_asr_triton_template/layers.py` | RMSNorm, LayerNorm, GELU, SiLU, Linear (tile-tuned), fused Linear+GELU, fused SwiGLU kernels |
| `glm_asr_triton_template/rope.py` | Rotary Position Embedding (RoPE) kernel |

Additional scripts we wrote:

| File | Description |
|------|-------------|
| `glm_asr_triton_template/kernel_tune.py` | Benchmarks kernel tile configurations (RMSNorm, LayerNorm, Linear+GELU, SwiGLU) |
| `glm_asr_triton_template/tile_tune.py` | Benchmarks tile configurations for `linear_kernel_tf32` across representative shapes |

### Running the Tuning Scripts

**`kernel_tune.py`** — sweeps tile configurations for all four kernels (RMSNorm, LayerNorm, fused Linear+GELU, fused SwiGLU) across representative GLM-ASR shapes and prints a ranked results table:

```bash
cd hw1-asr/glm_asr_triton_template
python kernel_tune.py
```

Results are saved to `results/kernel_tuning/kernel_tune_results.txt`.

**`tile_tune.py`** — sweeps 12 tile configurations for `linear_kernel_tf32` across three workload shapes (audio encoder attention, MLP projection, decoder GEMV) and prints a ranked latency table:

```bash
cd hw1-asr/glm_asr_triton_template
python tile_tune.py
```

Results are saved to `results/kernel_tuning/tile_tune_results.txt`.

---

## Setup

We use the **Triton** environment. Full setup instructions are in `GUIDE.md` and `README.md`.

### Cluster Interactive Session (Teaching Cluster)

To request an interactive GPU session on the teaching cluster:

```bash
srun -p Teaching --nodelist=saxa --gres=gpu:1g.18gb:1 --cpus-per-task=2 --mem=64G --pty bash
```

> **Note:** We ran with `--mem=64G` for faster loading of model weights and intermediate files. All scripts are compatible with the standard 16 GB MIG slice — the higher RAM allocation does **not** affect benchmark results, only loading speed.

**Quick setup (from repository root, one level above `hw1-asr/`):**

```bash
source utils/setup-triton.sh
```

**Verify baseline works:**

```bash
bash hw1-asr/benchmark.sh glm_asr_triton_example
```

**Run our implementation:**

```bash
bash hw1-asr/benchmark.sh glm_asr_triton_template
```

**Detailed component-level profiling:**

```bash
bash hw1-asr/benchmark_detailed.sh glm_asr_triton_template
```

**Custom audio clips:**

```bash
bash hw1-asr/benchmark.sh glm_asr_triton_template --audio hw1-asr/custom_dataset/Lecture.wav
bash hw1-asr/benchmark.sh glm_asr_triton_template --audio hw1-asr/custom_dataset/Nursery_Rhyme.wav
bash hw1-asr/benchmark.sh glm_asr_triton_template --audio hw1-asr/custom_dataset/Noisy_Env.wav
```

> **Note on accuracy for custom clips:** The course-provided `benchmark_student.py` has no `--reference` parameter and hardcodes the expected transcription for `test_audio.wav`. When a custom audio file is passed, the script still compares against that hardcoded text, so the reported accuracy will be near 0%. This is a limitation of the course-provided benchmark script, not a bug in our implementation — the **transcription output itself is correct**. To verify, inspect the printed `Transcription:` line in the benchmark output directly.

---

## Custom Dataset

Located in `hw1-asr/custom_dataset/`:

| File | Duration | Description |
|------|----------|-------------|
| `Lecture.wav` | 5.00 s | Cardiac physiology lecture (Khan Academy) |
| `Nursery_Rhyme.wav` | 5.00 s | Twinkle Twinkle Little Star |
| `Noisy_Env.wav` | 5.00 s | Speech in a noisy environment |

All clips are sampled at 16 kHz. Transcriptions were manually verified.

---

## Results Folder

All experimental results and analysis are in `hw1-asr/results/`:

| Subfolder | Contents |
|-----------|----------|
| `benchmarks/` | End-to-end benchmark results including custom dataset (`DATASET_RES.md`, `results.md`) |
| `fusion/` | Raw output from kernel fusion experiments (fused vs unfused variants) |
| `kernel_tuning/` | Tile size tuning results for `linear_kernel_tf32` and SwiGLU/Linear+GELU kernels |
| `analysis/` | Detailed analysis markdown files: ablation studies, FlashAttention design notes, layer verification, generate-v8 ablation |

---

## Environment

| Component | Version |
|-----------|---------|
| Python | 3.11 |
| PyTorch | 2.10.0+cu128 |
| Triton | 3.6.0 |
| CUDA | 12.8 |
| Hardware | NVIDIA H200 MIG 1g.18gb |
| Model | `zai-org/GLM-ASR-Nano-2512` (HuggingFace) |
