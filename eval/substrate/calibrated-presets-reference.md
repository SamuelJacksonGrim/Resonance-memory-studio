# Calibrated presets (fair invocation, 2026-09-06)

Panel shape: `{ dedup_hi, dedup_lo, field_minsim, constraint_gate, save_bind, dim }`.

**Do not reuse `fire-together-embedder-comparison.md`.** That run encoded every model PLAIN. Jina's 0.05–0.87 true-pair range and "nomic wins, keep it" as a *comparison* result are artifacts of wrong usage. Nomic remaining the default is still correct, but for different reasons — see `fire-together-embedder-fair-report.md`.

Dedup bands come from `eval/corpora/duplicates.jsonl` (restatements vs mid-band vs controls), never from theme-member pairs. Field numbers come from fire-together pairwise + kNN leak.

## nomic-embed-text-v1.5 (dim=768, Apache-2.0) — shipped default, PLAIN

Production `server.js` still sends unprefixed text for nomic (`embed-invoke.js`). That is nomic's *best* document-document geometry on this task; Qwen/jina selections now get their own prefixes so they are no longer the broken plain geometry.

```json
{
  "dedup_hi": 0.95,
  "dedup_lo": 0.88,
  "field_minsim": 0.70,
  "constraint_gate": 0.45,
  "save_bind": 0.25,
  "dim": 768
}
```

Dedup 0.95/0.88 are the existing shipped values, re-confirmed against ST-measured duplicates (hi-band min 0.963, mid 0.928–0.949, control/cross p95 0.556). Do not replace them with ST percentiles; the served GGUF is the one the 0.95/0.88 pair was tuned on.

**Shipped 2026-09-07:** `field_minsim: 0.70`. RM-00 golden held 27/31 case-for-case (sqlite + jsonl). Fire-together Related: true 14/15, near-miss 0, unrelated 0. Constraint rescue stays at 0.45.

## qwen3-embedding-0.6B (dim=1024, Apache-2.0) — now calibrated

Plain documents (Qwen's recipe). Not a recommended default — no associative win over nomic, 1024-dim store.

```json
{
  "dedup_hi": 0.95,
  "dedup_lo": 0.89,
  "field_minsim": 0.65,
  "constraint_gate": 0.55,
  "save_bind": 0.26,
  "dim": 1024
}
```

Safety: field 0.65, kNN true recall 0.93, 2 near-miss leaks, 0 unrelated. Do not copy nomic's 0.55 — on Qwen that gate leaks 13 near-miss edges.

## jina-embeddings-v5-text-nano-retrieval (dim=768, CC-BY-NC) — personal use, roles required

**These numbers assume `Document:` / `Query:` prefixes.** Plain jina is the broken geometry. Production does not send roles today.

```json
{
  "dedup_hi": 0.98,
  "dedup_lo": 0.84,
  "field_minsim": 0.55,
  "constraint_gate": 0.40,
  "save_bind": 0.15,
  "dim": 768
}
```

With roles, field 0.55 already has only 1 near-miss leak (true recall 0.93) because unrelated sits much lower (mean 0.19). License blocks a shipped default. Dim is 768, not 512.

## What these are not

- They are not a recommendation to switch embedder.
- `constraint_gate` is a weaker rescue floor than field minSim (production nomic: 0.45 vs 0.55). Kept ~0.10 below field.
- `save_bind` is a persist floor, deliberately looser (production 0.25).
