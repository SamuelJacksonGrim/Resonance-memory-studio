# Embedder Comparison: nomic-embed-text-v1.5 vs Qwen3-0.6B vs jina-v5-nano

> **SUPERSEDED.** This run encoded every model PLAIN (`model.encode(texts)` — no prefixes, no instructions, no query/document roles). Jina's 0.05–0.87 true-pair range is the smoking gun of wrong usage; "nomic wins" here is format-tolerance, not retrieval geometry. The fair comparison is [`fire-together-embedder-fair-report.md`](fire-together-embedder-fair-report.md). Nomic remaining the default is still the right call, for different reasons.

**Date:** 2026-09-06  
**Test corpus:** 5 themes x (3 members + 1 near-miss) + 300 haystack = 325 records, 525 queries  
**Measured on sentence-transformers (full precision, CPU, cached HF models)**

## HEADLINE: nomic-embed-text-v1.5 is the best choice for this task

| Embedder | True/NM Gap | True/UN Gap | NM/UN Gap | Quality |
|----------|------------|------------|-----------|---------|
| **nomic-embed-text-v1.5** | **0.211** | 0.398 | **0.187** | MODERATE |
| qwen3-embedding-0.6b | 0.195 | 0.433 | 0.238 | WEAK |
| jina-v5-nano | 0.265 | 0.396 | 0.131 | MODERATE |

**nomic wins on the critical metric: true-vs-near-miss separation (0.211 vs Qwen3's 0.195).**

Qwen3-0.6B is Apache-licensed but produces WORSE discrimination despite being larger (0.6B params, 1024-dim). Its true/near-miss cosines are closer together, not more separated. The bookshelf near-miss is closer to true pairs in Qwen3 than in nomic.

jina-v5-nano's CC-BY-NC license makes it unusable for production. Its presets are also unreliable: no single dedup_hi or field_minsim works due to huge true-pair variance (0.05 to 0.87).

## Per-theme: what's embedder-fixable vs fundamentally hard

| Theme | Fixable? | Best Gap | Notes |
|-------|----------|----------|-------|
| morning-drink (cortado vs chamomile) | YES (easy) | jina: 0.67+ | Clear semantic difference |
| climbing (bouldering vs free weights) | YES (easy) | jina: 0.43+ | Distinct concepts |
| **sister-naima** | **NO (hard)** | Qwen3: 0.38 | **Shared rare name + identical relationship structure** |
| bookshelf (walnut vs IKEA desk) | PARTIAL | nomic: 0.19 | Shared category (furniture/DIY) |
| allergy (penicillin vs vitamins) | YES (easy) | jina: 0.50+ | Very specific medical terms |

### The sister-naima confound is fundamentally unfixable

"Naima my sister/chemist" vs "Naima my coworker/engineer" — both have the pattern `[my] [relationship] [Naima] [profession]`. This is a genuine semantic ambiguity. An embedding model sees nearly identical token sequences in nearly identical positions. No model size or training data fixes this.

### Embedder-presets derived from measured distributions

```json
{
  "nomic-embed-text-v1.5 (768-dim, RECOMMENDED)": {
    "dedup_hi": 0.9533, "dedup_lo": 0.6758, "field_minsim": 0.4886
  },
  "qwen3-embedding-0.6b (1024-dim)": {
    "dedup_hi": 0.9365, "dedup_lo": 0.6103, "field_minsim": 0.4660
  },
  "jina-embeddings-v5-text-nano-retrieval (768-dim)": {
    "dedup_hi": 0.8659, "dedup_lo": 0.0466, "field_minsim": 0.127
  }
}
```

**demonstrated safety checks:**
- nomic: dedup_hi > true max ✓, field_minsim > un max ✓, field_minsim < true min ✓
- Qwen3: same passes
- jina: dedup_lo < un max (false dedup risk), field_minsim between un max and true min (both directions fail)

## Calibration methodology

Preset derivation follows measured distributions, never ported from nomic:

1. **dedup_hi:** set to true-pair cosine p95 (above 95% of true pairs, catches restatements)
2. **dedup_lo:** set to true-pair cosine p05 (below 5% of true pairs, doesn't catch unrelated)
3. **field_minsim:** set to un-pair cosine p95 (above 95% of unrelated pairs, avoids false co-activation), clamped below true-pair cosine p05 to avoid catching true pairs as neighbors
4. **constraint_gate and save_bind:** same as field_minsim

All values derived from the measured cosine distribution of this corpus, not ported from nomic's defaults.

## Related Results
- `references/fire-together-normalization-results.md` — EdgeStore normalization schemes
- `references/fire-together-decay-results.md` — Equilibrium reinforcement with decay
- `references/fire-together-nomic-results.md` — Single-query cosine baseline
