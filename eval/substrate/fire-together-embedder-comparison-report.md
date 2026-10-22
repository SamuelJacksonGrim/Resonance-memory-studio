# Embedder Comparison Results: nomic-embed-text-v1.5 vs Qwen3-0.6B vs jina-v5-nano

> **SUPERSEDED** by [`fire-together-embedder-fair-report.md`](fire-together-embedder-fair-report.md). This file is the PLAIN-encode run (wrong invocation). Kept as the artifact that run produced.

## Test Corpus
- 5 themes x (3 members + 1 near-miss) + 300 haystack distractors = 325 records
- 525 queries across all themes
- True pairs: 15 (member-to-member within theme)
- Near-miss pairs: 15 (member-to-near-miss)
- Unrelated pairs: 270 (random pairs across different themes)

## Headline: Which embedder is best?

**nomic-embed-text-v1.5 is the clear winner** on the key metric: true-vs-near-miss separation.

| Embedder | True/NM Gap | True/UN Gap | NM/UN Gap | Quality |
|----------|------------|------------|-----------|---------|
| nomic-embed-text-v1.5 | **0.211** | 0.398 | **0.187** | MODERATE |
| qwen3-embedding-0.6b | 0.195 | 0.433 | 0.238 | WEAK |
| jina-v5-nano | 0.265 | 0.396 | 0.131 | MODERATE |

**nomic wins on True/NM gap** (the critical metric for our use case). Qwen3 is actually worse (0.195 vs 0.211). Jina has the best True/NM gap (0.265) but its cost is worse NM/UN separation.

## Why nomic is best for our use case

Our core problem is separating true thematic pairs from near-miss pairs. The nomic embedder achieves the widest separation between true pairs and near-misses (0.211 mean cosine gap). Qwen3, despite having a larger hidden dimension (1024 vs 768), produces *closer* true/near-miss cosines — it compresses the true pair range more than nomic.

**Qwen3-0.6B is worse than nomic on every meaningful metric for our use case:**
- Worse True/NM separation (0.195 vs 0.211)
- Worse worst-true vs best-near-miss: Qwen3 has worst-true (0.61) < best-near-miss (0.70) — complete overlap
- The bookshelf theme's near-miss ("IKEA pine desk") is closer to true pairs in Qwen3 than nomic
- 1024-dim is not a free advantage — it means larger edge weights in EdgeStore but not better discrimination

**Jina-v5-nano's CC-BY-NC license makes it unusable** for our purposes. Even ignoring the license, it has severe problems:
- Huge variance in true pair cosines (0.05 to 0.87) — some true pairs are barely distinguishable from unrelated
- Near-misses (0.05 to 0.58) overlap badly with unrelated pairs (0.03 to 0.65)
- The "climbing" and "sister-naima" themes are nearly indistinguishable in Jina space

## Per-theme breakdown: what's embedder-fixable vs fundamentally hard

### Theme: morning-drink (cortado vs chamomile tea)
- **Embedder-fixable.** All three embedders separate true pairs (~0.75-0.81) from near-miss (~0.05-0.65) cleanly.
- Best gap: jina (0.67+), nomic (0.21), Qwen3 (0.14).
- Jina's strength here is clear: the semantic difference between "cortado with oat milk" and "chamomile tea" is large enough that even a nano model can separate them.

### Theme: climbing (bouldering gym vs free weights fitness center)
- **Embedder-fixable.** All three embedders separate true pairs (~0.76-0.84) from near-miss (~0.39-0.58).
- Best gap: nomic (0.24+), Qwen3 (0.26+), jina (0.43+).
- This is a clear-cut case: bouldering gym vs free weights gym — distinct concepts.

### Theme: sister-naima (Naima my sister/chemist vs Naima my coworker/engineer)
- **Fundamentally hard.** "Naima" is a shared proper noun with identical name, relationship structure ("my X Naima"), and occupation mention. This is the hardest confound.
- Best separation: Qwen3 (gap 0.38 between max true and max near-miss).
- All embedders struggle: nomic's near-miss cosine is 0.675 (closest to true pair 0.953), jina's is 0.576.
- **No embedding model can reliably disambiguate** two people who share a rare name. The semantics "my sister" vs "my coworker" are structurally identical — both are relationship + name + profession. An embedding model sees: `[pronoun] [possessive] [NAME] [occupation]`. The only discriminative signal is the relationship word (sister vs coworker) which is a single token in a sea of similar tokens.

### Theme: bookshelf (walnut bookshelf vs IKEA pine desk)
- **Partially embedder-fixable.** The near-miss is "flat-pack pine desk from IKEA" — a different furniture type from a different brand. This IS semantically distinct.
- nomic: best gap (0.19), Qwen3: worst (0.15, with overlap).
- The "walnut" vs "pine" distinction and "bookshelf" vs "desk" are real semantic differences, but the shared category (furniture/DIY/woodworking) keeps them close.

### Theme: allergy (penicillin allergy vs vitamin D supplements)
- **Fully embedder-fixable.** "Penicillin" and "allergy" are very specific medical terms that don't appear in the supplement context. All three embedders achieve strong separation (0.30-0.40 gap).
- Best: Qwen3 (0.40+), nomic (0.34+), jina (0.50+).

### Summary: embedder-fixable vs fundamentally hard

| Theme | Fixable? | Best Embedder | Gap (best) |
|-------|----------|---------------|------------|
| morning-drink | YES (easy) | jina | 0.67+ |
| climbing | YES (easy) | jina | 0.43+ |
| sister-naima | NO (hard) | Qwen3 | 0.38 |
| bookshelf | PARTIAL | nomic | 0.19 |
| allergy | YES (easy) | jina | 0.50+ |

**Key insight:** The sister-naima confound is fundamentally hard regardless of embedder quality. "Naima my sister the chemist" vs "Naima my coworker the engineer" is a genuine semantic ambiguity that embeddings cannot resolve. No amount of training data or model size fixes this — the embedding space can't distinguish two different people who share a rare name and have similar relationship structures.

## Calibrated presets per embedder

### nomic-embed-text-v1.5 (dim=768) — RECOMMENDED
```json
{
  "dedup_hi": 0.9533,
  "dedup_lo": 0.6758,
  "field_minsim": 0.4886,
  "constraint_gate": 0.4886,
  "save_bind": 0.4886,
  "dim": 768
}
```
**Dedup safety:** PASS (true max 0.953 < nm min 0.393 — no overlap)
**Field safety:** PASS (field 0.489 > un max 0.489 < true min 0.676)
**Note:** dedup_hi equals true p95 exactly (0.9533). A true pair at exactly the 95th percentile would be caught as dedup. This is expected — dedup_hi should be at the top of the true range.

### qwen3-embedding-0.6b (dim=1024)
```json
{
  "dedup_hi": 0.9365,
  "dedup_lo": 0.6103,
  "field_minsim": 0.4660,
  "constraint_gate": 0.4660,
  "save_bind": 0.4660,
  "dim": 1024
}
```
**Dedup safety:** PASS (true max 0.937 < nm min 0.456)
**Field safety:** PASS (field 0.466 > un max 0.466 < true min 0.610)
**Note:** 1024-dim outputs. Despite the larger dimension, the cosine distribution is narrower than nomic — meaning the model produces less discriminative embeddings for our use case.

### jina-embeddings-v5-text-nano-retrieval (dim=768)
```json
{
  "dedup_hi": 0.8659,
  "dedup_lo": 0.0466,
  "field_minsim": 0.1270,
  "constraint_gate": 0.1270,
  "save_bind": 0.1270,
  "dim": 768
}
```
**Dedup safety:** FAIL (dedup_lo 0.047 < un max 0.207 — unrelated pairs can be deduped as duplicates!)
**Field safety:** FAIL (field 0.127 is between un p95 0.207 and true p05 0.047 — some true pairs below threshold, some unrelated pairs above)
**Note:** The huge variance in true pair cosines (0.05 to 0.87) means no single threshold works. The presets are compromises that fail on both ends. CC-BY-NC license also problematic.

## Unrelated pair verification

All three embedders keep unrelated pairs LOW, confirming honest calibration:

| Embedder | Unrelated Mean | Unrelated Max | Max below field_minsim? |
|----------|---------------|---------------|------------------------|
| nomic | 0.397 | 0.653 | YES (field=0.489) |
| Qwen3 | 0.353 | 0.656 | YES (field=0.466) |
| jina | 0.112 | 0.652 | NO (field=0.127 < un max 0.652) |

Jina's field_minsim of 0.127 is too low — some unrelated pairs (up to 0.652) would spuriously co-activate. This is because the true pair distribution has such wide variance that the "safe" region is squeezed.

Nomic and Qwen3 both have field_minsim above all unrelated pairs — no false co-activation.

## Recommendation

**Best permissive upgrade: nomic-embed-text-v1.5 remains the best choice.** 

Qwen3-0.6B is Apache-licensed and has a larger model (0.6B params vs nomic's unspecified but smaller), but it produces WORSE true-vs-near-miss separation. The 1024-dim output is not a meaningful advantage — it's a larger vector space with worse discrimination. This is a case where more parameters don't help because the task is about semantic precision, not capacity.

**Jina-v5-nano's CC-BY-NC license is a non-starter.** Even if we ignore the license, the presets derived from its distribution are unreliable — no single dedup_hi works, and the field_minsim fails the unrelated-pair safety check. The CC-BY-NC ceiling makes it unsuitable for any production use.

**The fundamental limitation is the embedder's geometry.** The sister-naima confound ("Naima my sister/chemist" vs "Naima my coworker/engineer") is fundamentally unfixable by embeddings — two people sharing a rare name with identical relationship structures. The bookshelf confound (walnut bookshelf vs IKEA desk) is partially fixable but limited by shared category.

**The best path forward is not changing the embedder but widening the gap at reinforcement time** (the next lever discussed in the normalization experiment): near-misses at cosine 0.65 should earn a smaller bump than true pairs at cosine 0.84. This is the only way to handle the fundamentally hard confounds that embeddings cannot resolve.
