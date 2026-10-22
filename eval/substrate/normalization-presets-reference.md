# Calibrated Presets: EdgeStore Normalization Schemes

**Derived from equilibrium simulation results (2 years of use, 325 records, 300 haystack)**

## Problem
EdgeStore's stored weight has a 1.47x true/near-miss ratio at equilibrium. But `bonus()` reads it through `tanh(maxBonus)` which saturates for any weight > 3. Both true (~44) and near-miss (~30) hit the ceiling, collapsing the ratio to 1.00x.

## What works (honest recovery)

**Best honest scheme: divide-by-node-max**
- Recovers ~1.38x of the 1.47x stored weight ratio
- Formula: `bonus = tanh(stored_weight / node_max_edge_weight) * maxBonus`
- This normalizes each edge relative to the node's strongest edge
- Honest: doesn't invent separation, only reveals what's there in the weights

**Other honest schemes (less effective):**
- divide-by-node-mean: recovers ~1.17x
- z-score: recovers ~1.19x
- rank-based: recovers ~1.13x

**NOT honest (manufactures separation):**
- softmax with low tau: recovers 1.92-2.37x but concentrates bonus on one edge
- range normalization: recovers 1.01x (still saturated)

## Key finding
The tanh ceiling is a hard limit. No normalization that doesn't reduce the tanh argument below ~3 for all edges recovers the full 1.47x ratio. The gap between the stored weight ratio (1.47x) and the normalized bonus ratio (1.38x best case) is because tanh is still non-linear.

**Only lever that can widen the gap: similarity/rank-weighted reinforcement at save-time**, where a near-miss at cosine 0.65 earns a smaller bump than a true pair at cosine 0.84.

## Implementation sketch

```javascript
// For each node, track its maximum edge weight
function getNormalizedBonus(edgeStore, nodeId, targetId) {
  const storedWeight = edgeStore.weight(nodeId, targetId);
  
  // Find the node's maximum edge weight (across all neighbors)
  const neighbors = edgeStore.getNeighbors(nodeId);  // or equivalent
  let maxWeight = 0;
  for (const neighbor of neighbors) {
    maxWeight = Math.max(maxWeight, edgeStore.weight(nodeId, neighbor));
  }
  
  // Normalize and apply tanh
  const normWeight = maxWeight > 0 ? storedWeight / maxWeight : storedWeight;
  return Math.tanh(normWeight) * MAX_BONUS;
}
```

Note: `edgeStore` would need a `getNeighbors()` or `incident()` method to enumerate edges. Currently EdgeStore doesn't expose this in its public API.

## Related
- `eval-design` skill: evaluation design and corpus-based testing
- `eval/substrate/fire-together-normalization.js`: experiment script
- `eval/substrate/fire-together-normalization-results.md`: full results
- `eval/substrate/fire-together-embedder-comparison.md`: embedder comparison
