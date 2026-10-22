# EdgeStore Normalization Scheme Results

**Date:** 2026-09-06  
**Context:** EdgeStore's stored weight has a clean 1.47x true/near-miss ratio at equilibrium, but `bonus()` reads it through `tanh(maxBonus)` which saturates at any weight > 3. Both true (44) and near-miss (30) hit maxBonus = 0.3, collapsing the ratio to 1.00x. This experiment tests whether neighborhood-competitive normalization recovers the separation.

**Measured at session boundary after 2 years of equilibrium usage (104 weeks, 325 records, 300 haystack).**

## Scheme Results (normalized bonus values)

| Scheme | True Bonus | Near Bonus | Unrelated | True/Near Ratio |
|--------|-----------|-----------|-----------|----------------|
| Divide by node's max edge weight | 0.220 | 0.159 | 0.006 | **1.38x** |
| Divide by node's mean edge weight | 0.300 | 0.257 | 0.017 | 1.17x |
| Z-score within node's neighborhood | 0.292 | 0.246 | 0.172 | 1.19x |
| Rank-based (fraction of edges below) | 0.269 | 0.237 | 0.019 | 1.13x |
| Softmax (tau=0.1) | 0.061 | 0.026 | ~0 | 2.37x |
| Softmax (tau=0.5) | 0.068 | 0.035 | ~0 | 1.92x |
| Softmax (tau=1.0) | 0.072 | 0.038 | ~0 | 1.92x |
| Softmax (tau=2.0) | 0.083 | 0.041 | ~0 | 2.06x |
| Subtract node's min (range norm) | 0.300 | 0.298 | -0.070 | 1.01x |

## Baseline (current, no normalization)
- True: 0.300 (saturated) | Near: 0.300 (saturated) | **Ratio: 1.00x**

## Baseline (raw stored weights)
- True: 43.93 | Near: 29.87 | Unrelated: 1.00
- **True/Near ratio: 1.47x**

## Conclusions

1. **Divide-by-node-max recovers 1.38x** — closest to the true 1.47x. This is the strongest honest recovery.
2. **Divide-by-node-mean recovers 1.17x** — meaningful but compressed. The mean is inflated by true pairs.
3. **Z-score recovers 1.19x** — similarly compressed. Std is also driven by true pairs.
4. **Rank-based recovers 1.13x** — ordinal but compressed. The ranking preserves order but not ratio.
5. **Softmax with low tau recovers 1.92-2.37x** — artificially inflates by concentrating bonus on the strongest edge. NOT honest for Related:.
6. **Range normalization recovers NOTHING** — still saturated. Subtracting min doesn't help (41 vs 29 still > tanh threshold).

**The tanh ceiling is a hard limit.** No normalization that doesn't reduce the tanh argument below ~3 for all edges will recover the full 1.47x ratio. The gap between the stored weight ratio (1.47x) and the normalized bonus ratio (1.38x best case) is because tanh is still non-linear even at these normalized values.

**The only lever that can widen the gap is at reinforcement time:** a near-miss at cosine 0.65 should earn a smaller bump than a true pair at cosine 0.84. This is where real separation can be manufactured.

## Related Results
- `references/fire-together-embedder-comparison.md` — Embedder comparison (nomic vs Qwen3 vs Jina)
- `references/fire-together-decay-results.md` — Equilibrium reinforcement with decay
- `references/fire-together-nomic-results.md` — Single-query cosine baseline
