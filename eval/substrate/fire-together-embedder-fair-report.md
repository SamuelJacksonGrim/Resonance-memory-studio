# Fair embedder comparison — invoked correctly

**Date:** 2026-09-06
**Corpus:** fire-together (15 theme members + 5 labeled near-misses + 5 distractors + 300-seed-42 haystack, 525 queries); `eval/corpora/duplicates.jsonl` for dedup bands; 20 extra probes for confound classes the 5-theme set only samples once.
**Models:** `nomic-ai/nomic-embed-text-v1.5` (768, Apache), `Qwen/Qwen3-Embedding-0.6B` (1024, Apache), `jinaai/jina-embeddings-v5-text-nano-retrieval` (768, CC-BY-NC). Full precision, sentence-transformers, CPU. Cached under `eval/substrate/.cache-embedder-fair/`.
**Scripts:** `embedder_fair_run.py` (round 1), `embedder_fair_round2.py` (haystack / filter / field kNN / RRF / ensemble / weighted reinforce), `_ablate_qfilter.py` (relation-only vs fitted lexicon).

The prior `fire-together-embedder-comparison.py` encoded every model PLAIN. That measured format-tolerance. This run does not.

---

## Headline

**Swapping the embedder, even when each is used the way its authors specify, does not lift RM's associative ceiling.** The 1.47× true/near-miss cofire on nomic is a real number, not an artifact of wrong prefixes, and no candidate here beats it by enough to be a product change.

What *does* move the number, measured, with unrelated staying low:

| lever | what it is | true/nm cofire (haystack) | nm-in-top-5 | H1@5 | unrelated cofire |
|---|---|---|---|---|---|
| nomic PLAIN (production) | current geometry | **1.46×** | 0.64 | 1.00 | 0.0036 |
| nomic `search_document:` both sides | official-ish prefix, symmetric | 1.70× | 0.51 | 1.00 | 0.0021 |
| jina query/document roles | official jina | 1.54× | 0.59 | 1.00 | 0.0017 |
| qwen PLAIN | official docs-unprefixed | 1.52× | 0.58 | 0.998 | 0.0008 |
| RRF (nomic dense + BM25) | hybrid retrieval, ranks not scores | **2.01×** | 0.48 | 0.994 | 0.0039 |
| relation-only query filter | family/work/friend conflict | **1.78×** | 0.55 | **1.00** | 0.0036 |
| full category lexicon query filter | fitted to these 5 themes | 8.84× | 0.30 | 0.996 | 0.0036 |
| field minSim 0.55 → 0.70 (nomic Related:) | doc-doc gate, not cofire | — | **0 near-miss edges** in Related: | true-pair recall 0.93 | 0 unrelated edges |
| weighted reinforce (nomic, haystack, gate 0.55) | α × how far cosine sits above the gate | weight **6.18×** (tanh bonus 1.39×; node-max-norm 5.42×) | — | — | weight 0.028 |

The 8.84× is real on this corpus and does **not** inflate unrelated, but the category buckets (coffee/tea, climb/lift, shelf/desk, allergy/supplement) are fitted to the five themes. The **shippable** non-geometric signal is the closed-class relation conflict (family vs work vs friend), which is not theme-fitted: it cuts sister-Naima leak 0.78 → 0.25, lifts overall cofire 1.46 → 1.78, and does not miss a single H1.

---

## 1. Fair per-embedder comparison (each used correctly)

### How "correct" was invoked

| model | plain (prior run) | correct document-document (field / dedup / save-bind) | correct query-document (recall / cofire) |
|---|---|---|---|
| nomic | raw text | `search_document:` or `clustering:` on both | `search_query:` vs `search_document:` |
| Qwen3 | raw text | raw docs, or STS/cluster instruction on both | `prompt_name="query"` (`Instruct: Given a web search query…`) |
| jina | raw text | `prompt_name="document"` (`Document: `) | `prompt_name="query"` vs `document` |

Production RM (`server.js`) POSTs raw text to `/v1/embeddings` with **no prefixes**. So **nomic PLAIN is the production geometry**, not a wrong-usage condition. Prefixed nomic and role-tagged jina are product changes, not free upgrades.

### Document-document (the Related: / dedup geometry)

| tag | true mean | nm mean | un mean | true/nm AUROC | true/nm ratio | hard-sep? |
|---|---|---|---|---|---|---|
| **nomic__plain (production)** | 0.795 | 0.584 | 0.397 | **1.00** | 1.36 | **yes** (worst true 0.676 > best nm 0.676, barely) |
| nomic search_document both | 0.823 | 0.664 | 0.501 | 0.96 | 1.24 | no (Naima overlaps) |
| nomic clustering both | 0.891 | 0.767 | 0.606 | 0.99 | 1.16 | no — compressed |
| qwen plain / default query | 0.786 | 0.591 | 0.353 | 0.96 | 1.33 | no (bookshelf overlaps) |
| qwen cluster instruction | 0.810 | 0.592 | 0.351 | 0.98 | 1.37 | no |
| qwen STS both | 0.886 | 0.784 | 0.685 | 1.00 | 1.13 | compressed |
| jina PLAIN | 0.508 | 0.244 | 0.112 | 0.74 | 2.09 | no — H1 drops to 0.82, true p05=0.05. **The 0.05–0.87 smoking gun. Confirmed: wrong usage.** |
| **jina Document: both** | 0.743 | 0.443 | 0.187 | 0.99 | **1.68** | no (Naima + bookshelf overlap) |

**Jina used correctly collapses the variance.** True mean 0.51 → 0.74, H1 0.82 → 1.00, AUROC 0.74 → 0.99. The prior "jina is unusable" conclusion is an artifact and must not be reused. CC-BY-NC still blocks it as a shipped default.

**Nomic prefixes compress the space** (unrelated 0.40 → 0.50–0.61) and *hurt* true/nm AUROC. Clustering is the worst of these. Production PLAIN is nomic's best document-document geometry for this task.

Qwen's custom "distinguish shared names" cluster instruction does not beat nomic plain, and does not rescue Naima.

### Query-document with haystack (the cofire source, 325 records)

This is the 1.47× number. 25-record nm-in-top-5 was saturated (~95%) for everyone; ignore it.

| tag | H1@5 | nm-in-top-5 | cofire true/nm | un cofire |
|---|---|---|---|---|
| nomic__plain | 1.00 | 0.642 | **1.46** | 0.0036 |
| nomic search_document both | 1.00 | **0.507** | **1.70** | 0.0021 |
| nomic asymmetric query/doc | 1.00 | 0.857 | 1.17 | 0.0086 |
| qwen plain | 0.998 | 0.577 | 1.52 | 0.0008 |
| qwen default query prompt | 1.00 | 0.619 | 1.45 | 0.0018 |
| qwen cluster both | 1.00 | 0.608 | 1.50 | 0.0022 |
| jina Document: both | 1.00 | 0.594 | 1.49 | 0.0015 |
| jina asymmetric | 1.00 | 0.590 | 1.54 | 0.0017 |

Nomic's official *asymmetric* protocol is the **worst** cofire of the set (1.17×, nm-in-top-5 0.86). Symmetric `search_document:` on queries *and* docs is the one nomic prefix that helps retrieval-under-haystack. That is not how nomic tells you to do search, and it still loses on document-document Related: (field kNN at 0.55 leaks 12 near-miss edges vs 11 for plain, plus 3 unrelated).

**No embedder change is worth making for the associative bet.** The spread on haystack cofire is 1.17–1.70, production sits at 1.46, and the cleanest Related: geometry is still nomic plain.

---

## 2. Per-confound: embedder-fixable vs fundamental

### Fire-together themes (nomic plain, document-document)

| theme | true mean | nm mean | gap | overlap | haystack nm-in-top-5 on that theme's queries | fixable? |
|---|---|---|---|---|---|---|
| morning-drink (cortado vs chamomile) | 0.821 | 0.602 | 0.219 | no | 0.78 | **Partially.** Doc-doc separates; queries about "what I drink" still pull both. |
| climbing (bouldering vs free weights) | 0.852 | 0.650 | 0.202 | no | 0.72 | Same shape as drink. |
| **sister-naima** | 0.797 | 0.636 | 0.161 | no (barely: 0.703 vs 0.676) | 0.78 | **Name is lexical; identical-structure same-occupation is geometric-hard.** See probes. |
| bookshelf (walnut vs IKEA desk) | 0.747 | 0.580 | 0.166 | no | **0.88** | Partial. Shared furniture/DIY. Weakest true pair in the set (0.676). |
| allergy (penicillin vs vitamins) | 0.757 | 0.451 | 0.306 | no | **0.19** | **Yes.** Specific medical tokens. Already well separated. |

### Extra probes (not in the fire-together members)

These isolate classes the 5-theme set only samples once. Cosines are nomic plain / qwen plain / jina Document:.

| id | class | nomic | qwen | jina-doc | verdict |
|---|---|---|---|---|---|
| A1 | sister-Naima chemist vs coworker-Naima engineer (fire-together nm) | 0.65 | 0.65 | 0.46 | Occupations differ. Below nomic true-min. **Relation filter catches it.** |
| A2 | brother-Omar-dentist vs neighbor-Omar-dentist | **0.88** | **0.95** | **0.89** | **Fundamental.** Only the relation token differs. No embedder in this set separates it. |
| A3 | wife-Priya-painter vs colleague-Priya-painter | **0.92** | 0.91 | 0.90 | Same as A2. |
| A4 | "I call my sister Naima every Sunday" vs coworker paraphrase of the same template | 0.88 | 0.87 | 0.89 | Template copy + name. Fundamental without a relation signal. |
| Ap1/Ap2 | same name, same relation (restatement of one person) | 0.95 | 0.94–0.95 | 0.94–0.97 | Must stay high. All three keep it. |
| B1 | sister Naima chemist vs sister Layla chemist | 0.88 | 0.84 | 0.83 | Different person, same role. Geometry treats them as near-paraphrases. Needs an entity id, not a better embedder. |
| C1–C4 | shared category (the four non-Naima fire-together near-misses) | 0.46–0.68 | 0.47–0.71 | 0.33–0.52 | Embedder-separable in doc-doc, especially jina. |
| D1–D3 | restatements | 0.91–0.99 | 0.85–0.99 | 0.88–1.00 | All three keep these high. Jina PLAIN does not (D2 was 0.74; with Document: 0.98). |
| E1–E3 | unrelated | 0.29–0.40 | 0.26–0.31 | 0.05–0.19 | All stay low. Jina has the most headroom. |
| F1 | sister-chemist vs coworker-engineer, **no shared name** | 0.50 | 0.43 | 0.26 | Without the rare name, this is *not* a confound. |
| F2 | two sister-chemist paraphrases, no name | 0.95 | 0.95 | 0.94 | Structure + role is enough when the name isn't stealing the cosine. |

**The smoking-gun ablation (name mask, nomic plain, sister-Naima pairs):**

| pair | unmasked | proper-noun masked | shared-content masked |
|---|---|---|---|
| t3_m1 ↔ t3_m2 (true, both chemistry) | 0.953 | 0.831 | 0.754 |
| t3_m1 ↔ t3_nm (sister chemist vs coworker engineer) | 0.650 | **0.545** | 0.563 |
| t3_m2 ↔ t3_nm | 0.676 | **0.503** | 0.590 |
| t3_m3 ↔ t3_nm (Sunday call vs coworker) | 0.582 | **0.460** | 0.450 |

Masking `Naima` drops the near-miss into unrelated territory (~0.40–0.55). The residual gap is occupation/relation, which nomic *can* represent — the rare name was swamping it. Cost: t3_m2 ↔ t3_m3 (chemistry teacher vs Sunday call, same person) falls to 0.59, because those two facts mostly share the name. **Name-masking is the wrong mechanism** (it also unbinds facts about the same person). **Entity ids** are the right one: bind `Naima`+sister → E1 and `Naima`+coworker → E2, then Sunday-call and chemistry-teacher still share E1.

### What would actually resolve the hard cases

1. **Shared rare name + conflicting relation** (A1, A2, A3, A4): a closed-class relation lexicon (family / work / friend) applied as a retrieval penalty, or an entity linker that treats (name, relation) as the identity. Not a bigger embedder. Measured: relation-only filter, H1 stays 1.00, sister leak 0.78 → 0.25, unrelated unchanged.
2. **Same relation, different name, same occupation** (B1, two sisters who both teach chemistry): needs entity ids or user confirmation. Geometry will call them paraphrases. Correctly so, at the sentence level — they *are* paraphrases with one token swapped.
3. **Shared category, different instance** (cortado/tea, walnut/IKEA): already separable in nomic doc-doc; the leak is query underspecification ("what do I drink?"). RRF and a higher field minSim handle the Related: side. Co-activation leak on underspecified queries is the remaining associative cost of a 5-result window.

---

## 3. Calibrated presets

Derived from each model's own `duplicates.jsonl` (restatement vs mid-band vs controls) and fire-together pairwise (field / save-bind). Theme-member pairs are **not** restatements — the prior run's `dedup_hi = true p95` would merge distinct facts.

Panel shape: `{ dedup_hi, dedup_lo, field_minsim, constraint_gate, save_bind, dim }`.

### nomic-embed-text-v1.5 (768) — production, PLAIN

**Keep the shipped values.** They were tuned on `eval/duplicates` against the served GGUF, not this ST full-precision run.

```json
{ "dedup_hi": 0.95, "dedup_lo": 0.88, "field_minsim": 0.55, "constraint_gate": 0.45, "save_bind": 0.25, "dim": 768 }
```

ST-measured duplicates on this box: hi-band min 0.963 mean 0.978; mid-band 0.928–0.949; control/cross p95 0.556 max 0.692. Shipped 0.95/0.88 still sits in the right place (hi-band above 0.95, mid-band in the merge window, controls far below 0.88). Do not port the ST percentiles onto the GGUF.

**Measured field option, not shipped:** `field_minsim: 0.70`. On this corpus, Related: kNN at 0.70 keeps 28/30 true member-edges, **zero** near-miss edges, **zero** unrelated edges. At the shipped 0.55: 30/30 true, **11 near-miss leaks**. Raising it is a product decision that needs an RM-00 golden re-run (it will drop the weakest true pair, bookshelf m2↔m3 at 0.676). Compromise at 0.65: still 30/30 true, 4 near-miss leaks.

### qwen3-embedding-0.6B (1024) — Apache, now calibrated

Use **plain documents** (Qwen's own recipe). Query-side `prompt_name="query"` is optional and did not help cofire here.

```json
{ "dedup_hi": 0.95, "dedup_lo": 0.89, "field_minsim": 0.65, "constraint_gate": 0.55, "save_bind": 0.26, "dim": 1024 }
```

Duplicates: hi-band min 0.927 mean 0.968; mid 0.886–0.960. Field kNN at 0.65: true recall 0.93, 2 near-miss leaks, 0 unrelated. At 0.55: 13 near-miss leaks. Do not use nomic's 0.55 on Qwen — the geometry is different, and 0.55 on Qwen is as leaky as nomic's 0.55.

Not a recommended default. Apache-licensed, slightly better haystack nm-in-top-5 than nomic (0.58 vs 0.64), worse Related: overlap on bookshelf, 1024-dim store cost, and no win on the Naima-class probes.

### jina-embeddings-v5-text-nano-retrieval (768) — CC-BY-NC, calibrated for personal use

**These numbers assume `Document:` / `Query:` roles.** Plain jina is the broken geometry (dedup_hi would not even catch restatements: hi-band min 0.11). Production `server.js` does not send roles today, so selecting jina in the panel without a prefixing change would silently use the broken geometry.

```json
{ "dedup_hi": 0.98, "dedup_lo": 0.84, "field_minsim": 0.55, "constraint_gate": 0.40, "save_bind": 0.15, "dim": 768 }
```

Duplicates with Document: prefix: hi-band mean 0.99, mid 0.84–0.97. Field kNN at 0.55 already has only **1** near-miss leak (true recall 0.93) — the best Related: behaviour of the three at a 0.55-like gate, because unrelated sits much lower (mean 0.19). Cannot be a shipped default (CC-BY-NC). Dim in the panel was wrongly 512; the nano-retrieval checkpoint is 768.

---

## 4. Best achievable separation, and how

On this corpus, with unrelated held low:

**Related: (doc-doc field), nomic plain, minSim 0.70**
- True-pair recall 0.93, near-miss edges 0, unrelated edges 0.
- This is a threshold, not a new embedder. It is the single cheapest Related: cleanup.

**Co-activation source (query-doc, haystack), nomic plain + relation-only filter**
- Cofire 1.46× → 1.78×. Sister leak 0.78 → 0.25. H1 = 1.00. Unrelated unchanged.
- General: closed-class relation words, not a theme ontology.

**Co-activation source, nomic plain + BM25 RRF**
- Cofire 1.46× → 2.01×. nm-in-top-5 0.64 → 0.48. H1 0.994. Unrelated 0.0039.
- This is proposed/0003. It **changes primary ranking** if applied to the cosine top-k (I2). Using BM25 only to *gate reinforcement* (don't Hebbian-bump a pair BM25 disagrees with) would preserve I2 and still starve near-miss edges. Not implemented here; the number is the ceiling if you fuse ranks.

**Hebbian readout, nomic plain, haystack, 10 theme-cycled sessions**
- Equal bump (current EdgeStore): weight ratio 1.53×, tanh bonus **1.00×** (saturated), divide-by-node-max bonus **1.41×**. Replicates the Phase 2.3/2.4 finding.
- Cosine-gated bump (α × how far doc-doc cosine sits above field_minsim 0.55): weight **6.18×**, tanh bonus 1.39×, node-max-norm **5.42×**, unrelated weight 0.028.
- Gate at 0.70: near-miss weight goes to **0** (nothing above the gate earns a bump). That is a hard filter, not a smooth weight. Honest, and it starves the near-miss edges completely, at the cost of the 0.676 true pair.

**Compound (not jointly simulated, composition is obvious):** field minSim 0.70 so Related: never shows the near-miss, plus cosine-gated reinforce so Hebbian never wires it, plus a relation-conflict penalty so "sister Naima" queries don't retrieve the coworker. Unrelated stays low under each lever separately. Joint simulation on a larger store is the next measurement, not a claim.

**What does *not* work (measured):**
- Switching to Qwen3-0.6B or correctly-invoked jina as a default.
- Nomic `clustering:` / official asymmetric query-doc prefixes.
- Qwen STS or "distinguish shared names" instructions — they compress or don't help.
- Jaccard blend (0.7 cosine + 0.3 Jaccard) — Naima *shares* tokens, so Jaccard makes the confound worse, not better.
- nomic+jina rank ensemble — 1.40×, worse than nomic alone. Adding BM25 is what helps, and BM25+nomic is enough.
- Leave-one-theme-out rank-8 residual adapter: held-out AUROC looks fine on some folds and **inflates unrelated max to 0.79** (nomic plain, sister fold). A linear map over 4 themes of labels is not a product lever. Rejected.

---

## 5. Notes — tried, rejected, self-challenge

**Round 1 (correct invocation).** Confirmed the prior comparison was an artifact for jina (plain true-pair cosine 0.05–0.87; Document: prefix restores 0.74 mean, H1=1.00). Confirmed nomic prefixes do *not* help document-document discrimination. Confirmed Qwen used correctly is not better than nomic on the metric that matters (haystack cofire / Related: leak).

**Round 2, after looking at what round 1 did not measure.** 25-record nm-in-top-5 was saturated; re-ran cofire on the 300-haystack store. That recovered the 1.46× and showed nomic `search_document:` symmetric at 1.70× — the only prefix win, and it still loses Related:. Ran field kNN at 0.55/0.65/0.70 instead of only pairwise percentiles. Ran RRF on the haystack (2.01×). Ran query-conditioned entity penalty. Ran nomic+jina ensemble (rejected). Ran weighted reinforce on the haystack, not the 25-record toy (6.18× weight, unrelated 0.028).

**Self-challenge: is the 8.8× query-filter a fitted cheat?** Yes, partly. Ablation (`_ablate_qfilter.py`):

| filter | H1 | nm-in-top-5 | cofire ratio | sister leak | other-theme leaks |
|---|---|---|---|---|---|
| none (nomic plain) | 1.00 | 0.64 | 1.46 | 0.78 | drink 0.78, climb 0.72, shelf 0.88, allergy 0.19 |
| **relation-only** (family/work/friend) | **1.00** | 0.55 | **1.78** | **0.25** | unchanged |
| full 5-theme category lexicon | 0.996 | 0.30 | 8.84 | 0.25 | all drop to ~0.16–0.27 |

The 8.8× uses buckets that name the five near-misses. The 1.78× uses a closed class that would fire on Omar-dentist-brother vs Omar-dentist-neighbor without having seen that pair. Ship the 1.78×; treat 8.8× as an upper bound *if* RM grows a general category ontology, which it should not pretend to have.

**Self-challenge: would a learned projection work if I trained it harder?** The LoRA-style adapter already overfits unrelated (max 0.79 on a held-out theme). More steps on 15+15 pairs will not invent a direction the frozen embedder doesn't have. A real encoder fine-tune on thousands of (true, near-miss) pairs is a different project (Samuel's custom encoder), and the probe set says the A2/A3 class is a one-token distinction — that is what entity ids are for, not contrastive training.

**Self-challenge: cross-encoder / LLM rerank on :8080?** Not run. A pairwise reranker would almost certainly separate A2/A3 (a human does, in one glance). It is also a second model on the recall path, latency, and a ranking change (I2) unless used only to gate co-activation. The relation lexicon is the 80% of that idea that fits in the substrate with no new model.

**Self-challenge: is field minSim 0.70 overfit to 15 true pairs?** Yes, a little. The 0.676 bookshelf pair is the entire cost. A 50-theme corpus could put a true pair at 0.64 and a near-miss at 0.71. The *shape* (nomic plain hard-separates this set at ~0.70; production 0.55 leaks) is the finding. Do not ship 0.70 without a larger Related: measurement and an RM-00 pass.

**Production implication.** `server.js` still sends unprefixed text. Jina's calibrated preset is a lie unless roles are applied. Nomic should stay unprefixed. A prefixing switch in the embedder path, keyed on the panel's selected preset, is the productization step — not done here.

**Invariant check.** Nothing here changes primary ranking except the RRF row, which is flagged as 0003 and not recommended as a silent default. Field minSim, relation penalty on *co-activation*, and cosine-gated reinforce all leave cosine rank untouched (I2, I3).

---

## Artifacts

- `eval/substrate/embedder_fair_run.py` — round 1
- `eval/substrate/embedder_fair_round2.py` — round 2
- `eval/substrate/_ablate_qfilter.py` + `_ablate_qfilter.json` — relation-only vs fitted lexicon
- `eval/substrate/fire-together-embedder-fair.json` — per-mode pairwise, retrieval, probes, name-mask, adapter, 25-record reinforce
- `eval/substrate/fire-together-embedder-fair-round2.json` — haystack cofire, qfilter, field kNN, RRF, ensemble, haystack reinforce
- `eval/substrate/.cache-embedder-fair/` — embeddings (gitignored if `eval/substrate/.cache/` is; this sibling may need a gitignore line)
- `eval/substrate/calibrated-presets-reference.md` — panel-shaped objects
