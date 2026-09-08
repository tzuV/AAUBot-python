# Compute Budget

Estimates for running the AAUBot RAG system smoothly under the target load
of ~70 concurrent students. The LLM is the dominant cost; the embedding
model and ChromaDB are small by comparison.

All VRAM figures are per-GPU and assume vLLM serving on a single GPU. "Q4"
means 4-bit quantization (AWQ / GPTQ / Q4_K_M), which roughly halves quality
versus FP16/BF16 while cutting weight memory ~4x.

---

## 1. Where the VRAM goes

A vLLM deployment divides GPU memory into three buckets:

1. **Model weights** — fixed, loaded once. Scales with total params, not
   active params (matters for MoE: all experts live in VRAM even if only a
   few fire per token).
2. **KV cache** — grows with concurrent requests × sequence length. vLLM's
   PagedAttention allocates this dynamically from whatever VRAM the weights
   leave free. This is the lever for the 70-student target.
3. **Activations + overhead** — small, a few hundred MB to ~1 GB.

Embedding model: add ~2–4 GB (FP16) or <1 GB (INT4) on the same GPU.
ChromaDB: CPU RAM, negligible VRAM. System RAM: a few GB.

The binding constraint is: **more weights → less KV cache → fewer
concurrent students.** Quantization (Q4) is how you buy back concurrency.

### KV cache rule of thumb

For a 7B-class model at FP16 KV, one token of one request costs roughly
~1–2 MB. A 2 KB context (prompt + retrieved docs + answer) per student means
~2–4 MB × 2048 tokens ≈ 2–4 GB for a *single* long request, but vLLM batches
them in paged blocks. Practically, on a 24 GB GPU:

| Quant | Weights | Free for KV cache | Concurrent ~2 KB requests |
|-------|---------|-------------------|----------------------------|
| FP16  | ~14 GB  | ~8–9 GB           | ~15–25                      |
| Q4    | ~4–6 GB | ~16–18 GB         | ~70–100+                    |

So **FP16 on 24 GB likely can't hit 70 concurrent**; **Q4 on 24 GB can.**
This is the core trade-off in the budget.

---

## 2. Tiers: 3B / 7B / 11B

### 3B tier — dense

- Weights: ~6 GB (FP16), ~1.5–2 GB (Q4)
- Plenty of VRAM left for KV cache even on a 12 GB card
- Handles 70 concurrent easily on most GPUs
- Trade-off: weaker reasoning and code understanding; may give shallower or
  less accurate answers on tricky Python questions
- Good fit if GPU VRAM is tight (<16 GB) or latency must be very low

### 7B tier — dense (recommended default)

- Weights: ~14 GB (FP16), ~3.5–6 GB (Q4)
- On 24 GB at Q4: ~16–18 GB free for KV cache → comfortably 70+ concurrent
- On 24 GB at FP16: only ~8–9 GB for KV cache → ~15–25 concurrent, **misses
  the target** unless you quantize or add a second GPU
- Best balance of answer quality and throughput for a coding Q&A bot
- The sweet spot for a single 24 GB GPU (e.g. RTX 4090, A10, L4 24GB)

### 11B tier — dense

- Weights: ~22 GB (FP16), ~5.5–7 GB (Q4)
- FP16 barely fits on 24 GB with no room for KV cache — **not viable single-GPU
  at FP16**
- At Q4 on 24 GB: ~14–16 GB free for KV cache → ~60–90 concurrent, marginal
  for the 70 target; safer on a 32–48 GB GPU (A100 40GB, etc.)
- Higher answer quality, but the VRAM/concurrency margin is thin on 24 GB

### MoE note (important)

MoE models look small by *active* params but VRAM is driven by *total* params,
because every expert must be loaded even if only a few fire per token.

- A "30B-A3B" MoE (3B active, 30B total): needs ~60 GB FP16 / ~15–20 GB Q4
  just for weights — comparable to a 30B dense model for memory purposes
- At Q4 on 24 GB it fits but leaves little KV cache headroom, so concurrency
  is actually *lower* than a 7B dense at Q4
- MoE wins on quality-per-active-param and on raw token throughput (fewer
  active FLOPs), but for a fixed single-GPU VRAM budget it usually costs more
  memory per concurrent request than a smaller dense model
- Conclusion for this project: **MoE only makes sense if you have 32+ GB
  VRAM.** On 24 GB or less, a dense 7B at Q4 beats a 30B-A3B MoE on the
  70-student throughput target.

---

## 3. Model candidates

Ranked by fit for a Python-programming Q&A RAG bot, focused on the 3B–11B
range. Licenses are current as of the model's latest release; verify before
deployment.

### Dense, ~3B

| Model | Params | FP16 / Q4 VRAM | Notes |
|-------|--------|----------------|-------|
| Qwen3 4B | 4B | ~8 GB / ~2.5 GB | Strong multilingual incl. coding; Apache 2.0 |
| Phi-4 mini | 3.8B | ~7.6 GB / ~2 GB | Excellent reasoning for size; MIT |
| Gemma 3 4B | 4B | ~8 GB / ~2.5 GB | Solid general; Google license |
| Llama 3.2 3B | 3B | ~6 GB / ~2 GB | Broadly available; Llama license |

### Dense, ~7B

| Model | Params | FP16 / Q4 VRAM | Notes |
|-------|--------|----------------|-------|
| Qwen3 7B / 8B | 7–8B | ~14–16 GB / ~4–5 GB | **Top pick** — best coding/multilingual in this class; Apache 2.0 |
| Llama 3.1 8B | 8B | ~16 GB / ~5 GB | Strong general-purpose; Llama license |
| Mistral 7B v0.3 | 7B | ~14 GB / ~4 GB | Mature, well-supported; Apache 2.0 |
| Gemma 3 9B | 9B | ~18 GB / ~5.5 GB | High quality, slightly heavier |

### Dense, ~11–14B

| Model | Params | FP16 / Q4 VRAM | Notes |
|-------|--------|----------------|-------|
| Qwen3 14B | 14B | ~28 GB / ~8 GB | Excellent quality; needs Q4 on 24 GB, better on 32–48 GB; Apache 2.0 |
| Gemma 3 12B | 12B | ~24 GB / ~7 GB | Near the 24 GB ceiling at FP16; Q4 recommended |
| Mistral Nemo 12B | 12B | ~24 GB / ~7 GB | Built with NVIDIA; Apache 2.0 |

### MoE, small-active-param (only if 32+ GB VRAM)

| Model | Total / Active | FP16 / Q4 VRAM | Notes |
|-------|----------------|----------------|-------|
| Qwen3 30B-A3B | 30B / 3B active | ~60 GB / ~15–20 GB | 3B active but full 30B in VRAM; needs Q4 + ≥24 GB |
| MiniMax-01 (MoE) | 456B / 4.6B active | very high | Too large for single-GPU; multi-GPU only |

### Embedding models (same GPU)

| Model | VRAM (FP16) | Notes |
|-------|-------------|-------|
| bge-m3 | ~2–4 GB | Multilingual, strong on code/docs; recommended |
| e5-large-v2 | ~1.3 GB | Lighter, English-focused |
| bge-small-en-v1.5 | ~0.5 GB | Smallest, fine for English-only Python docs |

---

## 4. Recommendation

For a single 24 GB GPU serving ~70 concurrent students:

- **Default: Qwen3 8B (dense, Q4/AWQ)** + bge-m3 embedding.
  Weights ~5 GB, embedding ~2–4 GB, leaving ~14–16 GB for KV cache — enough
  for 70+ concurrent ~2 KB requests.
- **If VRAM is 12–16 GB:** drop to Qwen3 4B or Phi-4 mini (dense, Q4) +
  bge-small; still hits the concurrency target, lower answer quality.
- **If VRAM is 32–48 GB (A100 40GB etc.):** Qwen3 14B (dense, Q4) for better
  answers, or Qwen3 30B-A3B MoE (Q4) if you want MoE throughput and have
  headroom.
- **Avoid FP16 for any tier above 3B on 24 GB** — the KV cache is too
  squeezed to reach 70 concurrent.

### GPU sizing by tier (recommended min)

| Tier | Min GPU VRAM (Q4, single GPU) | Example GPUs |
|------|-------------------------------|--------------|
| 3B dense | 8–12 GB | RTX 3060 12GB, L4 |
| 7B dense | 16–24 GB | RTX 4090, A10, L4 24GB |
| 11–14B dense | 24–48 GB | A100 40GB, RTX 6000 Ada |
| MoE (30B-A3B) | 24–48 GB | A100 40GB+ |

### Throughput caveat

These are static VRAM estimates. The real concurrency ceiling depends on
request length (prompt + retrieved context + answer), vLLM's
`--max-model-len`, `--gpu-memory-utilization`, and batching efficiency. Before
signing off on a GPU size, run a load test with realistic ~2 KB requests at
70 concurrent and watch the KV cache. The numbers above are a starting
budget, not a guarantee.
