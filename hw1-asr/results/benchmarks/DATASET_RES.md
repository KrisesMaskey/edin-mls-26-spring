# Section 6.2 — Custom Dataset Benchmark Results

> Hardware: NVIDIA H200 MIG 1g.18gb | Protocol: 1 warmup + 3 timed runs

## End-to-End Latency Table

| Audio | Duration (s) | Baseline (ms) | Ours (ms) | Speedup | Transcription |
|---|---|---|---|---|---|
| test_audio.wav | 3.50 | 1474.8 ± 0.2 | 460.3 ± 0.3 | 3.20× | "Concord returned to its place amidst the tents." |
| Lecture.wav | 5.00 | 2103.7 ± 0.4 | 603.9 ± 8.7 | 3.48× | "So what you're looking at is one of the most amazing organs in your body." |
| Nursery_Rhyme.wav | 5.00 | 1204.6 ± 0.2 | 412.7 ± 1.6 | 2.92× | "Twinkle, twinkle, little star." |
| Noisy_Env.wav | 5.00 | 2107.1 ± 0.2 | 617.4 ± 1.8 | 3.41× | "We have an exhibit here called Speaking and Listening in a Noisy World." |

---

## Raw Output

### Lecture.wav — Baseline (`glm_asr_triton_example`)
- Time: 2103.7ms ± 0.4ms | Tokens: 18 | Speed: 116.87ms/token
- Transcription: "So what you're looking at is one of the most amazing organs in your body."

### Lecture.wav — Ours (`glm_asr_triton_template`)
- Time: 603.9ms ± 8.7ms | Tokens: 18 | Speed: 33.55ms/token
- Transcription: "So what you're looking at is one of the most amazing organs in your body." ✅

### Nursery_Rhyme.wav — Baseline (`glm_asr_triton_example`)
- Time: 1204.6ms ± 0.2ms | Tokens: 7 | Speed: 172.08ms/token
- Transcription: "Twinkle twinkle little star"

### Nursery_Rhyme.wav — Ours (`glm_asr_triton_template`)
- Time: 412.7ms ± 1.6ms | Tokens: 10 | Speed: 41.27ms/token
- Transcription: "Twinkle, twinkle, little star." ✅

### Noisy_Env.wav — Baseline (`glm_asr_triton_example`)
- Time: 2107.1ms ± 0.2ms | Tokens: 18 | Speed: 117.06ms/token
- Transcription: "We have an exhibit here called Speaking and Listening in a Noisy World."

### Noisy_Env.wav — Ours (`glm_asr_triton_template`)
- Time: 617.4ms ± 1.8ms | Tokens: 18 | Speed: 34.30ms/token
- Transcription: "We have an exhibit here called Speaking and Listening in a Noisy World." ✅
