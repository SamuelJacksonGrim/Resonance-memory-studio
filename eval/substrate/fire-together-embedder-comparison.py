#!/usr/bin/env python3
"""
Embedder comparison: nomic-embed-text-v1.5 vs Qwen3-Embedding-0.6B vs jina-v5-nano.

Measures:
  1. Cosine separation: true pairs vs near-miss vs unrelated
  2. Calibrated tuning presets per embedder
  3. Honest verification: unrelated pairs stay low, hard confounds flagged

Runs sentence-transformers locally (full precision). Uses cached models.
"""
import json
import os
import sys
import math

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ── Load corpus ──────────────────────────────────────────────────────────────
corpus_path = os.path.join(os.path.dirname(__file__), "..", "corpora", "fire-together-corpus.json")
corpus = json.load(open(corpus_path))
records = corpus["records"]
queries = corpus["queries"]
themes = corpus["themes"]
true_pairs = corpus["true_pairs"]
near_miss_pairs = corpus["near_miss_pairs"]
unrelated_pairs = corpus["unrelated_pairs"]

# Map record id -> record text
record_text = {r["id"]: r["text"] for r in records}

# ── Embedders ────────────────────────────────────────────────────────────────

# We need to load models. Since sentence-transformers may not be installed
# in the global env, we'll try to import it and use it.
# The models are cached locally, so we can use their paths directly.

QWEN_PATH = r"C:\Users\spamw\.cache\huggingface\hub\models--Qwen--Qwen3-Embedding-0.6B\snapshots\97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
JINA_PATH = r"C:\Users\spamw\.cache\huggingface\hub\models--jinaai--jina-embeddings-v5-text-nano-retrieval\snapshots\ac5d898c8d382b17167c33e5c8af644a3519b47d"
NOMIC_PATH = None  # loaded from HF

# We'll use sentence-transformers for loading. Check if available.
try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False
    print("sentence-transformers not available in this env — will use pytorch directly")

# ── Load models ──────────────────────────────────────────────────────────────

def load_models():
    """Load all three embedders. Return dict {name: model_or_config}."""
    models = {}
    
    # Nomic: loaded from HF hub
    try:
        from sentence_transformers import SentenceTransformer
        nomic = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", device="cpu")
        models["nomic-embed-text-v1.5"] = nomic
        print("Loaded nomic-embed-text-v1.5 (768-dim)")
    except Exception as e:
        print(f"Failed to load nomic: {e}")
    
    # Qwen3: loaded from local path (safetensors)
    try:
        from sentence_transformers import SentenceTransformer
        qwen = SentenceTransformer(QWEN_PATH, device="cpu")
        models["qwen3-embedding-0.6b"] = qwen
        print(f"Loaded Qwen3-Embedding-0.6B (1024-dim)")
    except Exception as e:
        print(f"Failed to load Qwen3: {e}")
    
    # Jina v5 nano: has GGUF files — need to load via native Jina interface
    # or via ONNX. Let's try sentence-transformers first, fallback to ONNX.
    try:
        from sentence_transformers import SentenceTransformer
        jina = SentenceTransformer(JINA_PATH, device="cpu")
        models["jina-v5-nano"] = jina
        print(f"Loaded jina-embeddings-v5-text-nano-retrieval (768-dim)")
    except Exception as e:
        print(f"Failed to load Jina via ST: {e}")
        # Fallback: try ONNX
        try:
            onnx_path = os.path.join(JINA_PATH, "onnx")
            if os.path.exists(os.path.join(onnx_path, "model.onnx")):
                print("Found ONNX files, but ONNX runtime not available for Jina")
        except:
            pass
    
    return models

# ── Embed texts ──────────────────────────────────────────────────────────────

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def embed_texts(model, texts):
    """Embed texts and return list of embeddings."""
    if model is None:
        return []
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True, device="cpu")
    if hasattr(embeddings, "tolist"):
        return embeddings.tolist()
    return [list(e) for e in embeddings]

# ── The experiment ───────────────────────────────────────────────────────────

def run_experiment(models):
    """
    For each embedder:
    1. Embed all records + queries
    2. Compute pairwise cosine for true/near/unrelated pairs
    3. Derive calibrated presets
    """
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*70}")
        print(f"EMBEDDER: {name}")
        print(f"{'='*70}")
        
        # Determine output dim
        sample_emb = model.encode(["test"], show_progress_bar=False, normalize_embeddings=True, device="cpu")
        if hasattr(sample_emb, "tolist"):
            dim = len(sample_emb.tolist()[0]) if isinstance(sample_emb.tolist()[0], list) else len(sample_emb)
        else:
            dim = len(sample_emb[0])
        print(f"  Output dimension: {dim}")
        
        # Embed all records
        record_texts = [r["text"] for r in records]
        print(f"  Embedding {len(records)} records...")
        record_embs = embed_texts(model, record_texts)
        
        # Build id -> embedding map
        id2emb = {}
        for r, emb in zip(records, record_embs):
            id2emb[r["id"]] = emb
        
        # ── Measure pairwise cosines ──
        true_cosines = []
        for pair in true_pairs:
            emb_a = id2emb.get(pair["a"])
            emb_b = id2emb.get(pair["b"])
            if emb_a and emb_b:
                true_cosines.append(cosine(emb_a, emb_b))
        
        nm_cosines = []
        for pair in near_miss_pairs:
            emb_a = id2emb.get(pair["a"])
            emb_b = id2emb.get(pair["b"])
            if emb_a and emb_b:
                nm_cosines.append(cosine(emb_a, emb_b))
        
        un_cosines = []
        for pair in unrelated_pairs:
            emb_a = id2emb.get(pair["a"])
            emb_b = id2emb.get(pair["b"])
            if emb_a and emb_b:
                un_cosines.append(cosine(emb_a, emb_b))
        
        print(f"\n  Pairwise cosine distributions:")
        print(f"  {'Type':<20} | {'Mean':>8} | {'Median':>8} | {'Std':>8} | {'Min':>8} | {'Max':>8} | {'N':>5}")
        print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*5}")
        
        for label, cosines in [("True pairs", true_cosines), ("Near-miss pairs", nm_cosines), ("Unrelated pairs", un_cosines)]:
            cosines_sorted = sorted(cosines)
            n = len(cosines)
            mean = sum(cosines) / n
            median = cosines_sorted[n // 2]
            std = math.sqrt(sum((c - mean) ** 2 for c in cosines) / n)
            mn = cosines_sorted[0]
            mx = cosines_sorted[-1]
            print(f"  {label:<20} | {mean:>8.4f} | {median:>8.4f} | {std:>8.4f} | {mn:>8.4f} | {mx:>8.4f} | {n:>5}")
        
        # ── Derive calibrated presets ──
        # dedup_hi / dedup_lo: measured from record-to-record cosine distribution
        # field_minsim: measured from true pair vs unrelated pair separation
        # constraint_gate: same as field_minsim (both use same signal)
        # save_bind: same as field_minsim
        # dim: model's output dim
        
        # For dedup bands: look at the distribution of record-to-record cosines
        # Dedup should catch true-theme-member pairs (high cosine) as restatements
        # while not flagging unrelated pairs
        
        # Find the gap between true pairs and near-misses
        true_sorted = sorted(true_cosines)
        nm_sorted = sorted(near_miss_pairs, key=lambda p: cosine(id2emb[p["a"]], id2emb[p["b"]]))
        
        # Best separation point: where true pairs are all above and unrelated all below
        # Use percentile analysis
        true_p50 = true_sorted[len(true_sorted) // 2]
        true_p95 = true_sorted[int(len(true_sorted) * 0.95)]
        true_p05 = true_sorted[int(len(true_sorted) * 0.05)]
        
        nm_sorted_cos = sorted(cosine(id2emb[p["a"]], id2emb[p["b"]]) for p in near_miss_pairs)
        nm_p50 = nm_sorted_cos[len(nm_sorted_cos) // 2]
        nm_p95 = nm_sorted_cos[int(len(nm_sorted_cos) * 0.95)]
        
        un_sorted = sorted(un_cosines)
        un_p95 = un_sorted[int(len(un_sorted) * 0.95)]
        un_p50 = un_sorted[int(len(un_sorted) * 0.5)]
        un_p05 = un_sorted[int(len(un_sorted) * 0.05)]
        
        # The "dedup_hi" should be above the highest true-pair cosine to avoid
        # false positive dedup, but below the highest near-miss cosine if possible.
        # dedup_lo should be below the lowest true-pair cosine.
        dedup_hi = true_p95  # above 95% of true pairs
        dedup_lo = true_p05  # below 5% of true pairs
        # Clamp to reasonable range
        dedup_hi = min(max(dedup_hi, 0.5), 0.99)
        dedup_lo = min(max(dedup_lo, 0.1), 0.95)
        
        # Field minSim: should be above the highest unrelated pair cosine
        # to avoid false co-activation, but below the lowest true-pair cosine
        field_minsim = un_p95  # above 95% of unrelated pairs
        # Also ensure it's below the lowest true-pair cosine
        field_minsim = min(field_minsim, true_p05)
        field_minsim = max(field_minsim, 0.1)  # don't go too low
        
        # Save bind: same as field_minsim
        save_bind = field_minsim
        
        # Constraint gate: same as field_minsim (both use same signal)
        constraint_gate = field_minsim
        
        # Store results
        results[name] = {
            "dim": dim,
            "presets": {
                "dedup_hi": round(dedup_hi, 4),
                "dedup_lo": round(dedup_lo, 4),
                "field_minsim": round(field_minsim, 4),
                "constraint_gate": round(constraint_gate, 4),
                "save_bind": round(save_bind, 4),
            },
            "separation": {
                "true_mean": round(sum(true_cosines) / len(true_cosines), 4),
                "true_median": round(sorted(true_cosines)[len(true_cosines) // 2], 4),
                "nm_mean": round(sum(nm_cosines) / len(nm_cosines), 4),
                "nm_median": round(nm_sorted_cos[len(nm_sorted_cos) // 2], 4),
                "un_mean": round(sum(un_cosines) / len(un_cosines), 4),
                "un_median": round(sorted(un_cosines)[len(un_cosines) // 2], 4),
                "true_vs_nm_gap": round(
                    (sum(true_cosines) / len(true_cosines)) - (sum(nm_cosines) / len(nm_cosines)), 4
                ),
                "true_vs_un_gap": round(
                    (sum(true_cosines) / len(true_cosines)) - (sum(un_cosines) / len(un_cosines)), 4
                ),
                "nm_vs_un_gap": round(
                    (sum(nm_cosines) / len(nm_cosines)) - (sum(un_cosines) / len(un_cosines)), 4
                ),
            },
            "true_cosines": true_cosines,
            "nm_cosines": nm_cosines,
            "un_cosines": un_cosines,
            "per_pair": {},  # detailed per-pair analysis
        }
        
        # Per-pair analysis: identify which near-misses are closest to true pairs
        for pair in near_miss_pairs:
            cos = cosine(id2emb[pair["a"]], id2emb[pair["b"]])
            results[name]["per_pair"][f"{pair['a']}_{pair['b']}"] = {
                "cosine": round(cos, 4),
                "text_a": record_text.get(pair["a"], "")[:50],
                "text_b": record_text.get(pair["b"], "")[:50],
            }
        
        # Check: are true pairs separated from near-misses?
        best_nm = max(nm_cosines)
        worst_true = min(true_cosines)
        separation_holds = worst_true > best_nm
        overlap = sum(1 for t in true_cosines for n in nm_cosines if abs(t - n) < 0.05)
        
        print(f"\n  SEPARATION ANALYSIS:")
        print(f"  True pairs:    mean={true_p50:.4f} median={true_p50:.4f}")
        print(f"  Near-miss:     mean={nm_p50:.4f} median={nm_p50:.4f}")
        print(f"  Unrelated:     mean={un_p50:.4f} median={un_p50:.4f}")
        print(f"  True-vs-NM gap: {results[name]['separation']['true_vs_nm_gap']:.4f}")
        print(f"  True-vs-UN gap: {results[name]['separation']['true_vs_un_gap']:.4f}")
        print(f"  NM-vs-UN gap:   {results[name]['separation']['nm_vs_un_gap']:.4f}")
        print(f"  Worst true ({worst_true:.4f}) > Best near-miss ({best_nm:.4f}): {separation_holds}")
        print(f"  Overlap (cos diff < 0.05): {overlap} pairs")
        
        # Derive field minSim: should be between true and unrelated
        field_min_sim = un_p95 if un_p95 < true_p05 else true_p05 - 0.01
        field_min_sim = max(field_min_sim, 0.1)
        results[name]["presets"]["field_minsim"] = round(field_min_sim, 4)
    
    return results

# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os  # needed for path ops
    
    print("=" * 70)
    print("EMBEDDER COMPARISON: nomic vs Qwen3-0.6B vs jina-v5-nano")
    print("Corpus: 5 themes x (3 members + 1 near-miss) + 300 haystack")
    print("=" * 70)
    
    # Load models
    models = load_models()
    
    if not models:
        print("\nERROR: No models loaded. Install torch and sentence-transformers:")
        print("  pip install torch sentence-transformers")
        sys.exit(1)
    
    # Run experiment
    results = run_experiment(models)
    
    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "fire-together-embedder-comparison.json")
    
    # Clean up per-pair data for output (keep only summary)
    output_results = {}
    for name, r in results.items():
        output_results[name] = {
            "presets": r["presets"],
            "separation": r["separation"],
            "dim": r["dim"],
            "num_true": len(r["true_cosines"]),
            "num_near_miss": len(r["nm_cosines"]),
            "num_unrelated": len(r["un_cosines"]),
            "true_cosine_hist": {
                "p05": round(sorted(r["true_cosines"])[int(len(r["true_cosines"]) * 0.05)], 4),
                "p25": round(sorted(r["true_cosines"])[int(len(r["true_cosines"]) * 0.25)], 4),
                "p50": round(sorted(r["true_cosines"])[int(len(r["true_cosines"]) * 0.5)], 4),
                "p75": round(sorted(r["true_cosines"])[int(len(r["true_cosines"]) * 0.75)], 4),
                "p95": round(sorted(r["true_cosines"])[int(len(r["true_cosines"]) * 0.95)], 4),
            },
            "near_miss_cosine_hist": {
                "p05": round(sorted(r["nm_cosines"])[int(len(r["nm_cosines"]) * 0.05)], 4),
                "p25": round(sorted(r["nm_cosines"])[int(len(r["nm_cosines"]) * 0.25)], 4),
                "p50": round(sorted(r["nm_cosines"])[int(len(r["nm_cosines"]) * 0.5)], 4),
                "p75": round(sorted(r["nm_cosines"])[int(len(r["nm_cosines"]) * 0.75)], 4),
                "p95": round(sorted(r["nm_cosines"])[int(len(r["nm_cosines"]) * 0.95)], 4),
            },
            "unrelated_cosine_hist": {
                "p05": round(sorted(r["un_cosines"])[int(len(r["un_cosines"]) * 0.05)], 4),
                "p25": round(sorted(r["un_cosines"])[int(len(r["un_cosines"]) * 0.25)], 4),
                "p50": round(sorted(r["un_cosines"])[int(len(r["un_cosines"]) * 0.5)], 4),
                "p75": round(sorted(r["un_cosines"])[int(len(r["un_cosines"]) * 0.75)], 4),
                "p95": round(sorted(r["un_cosines"])[int(len(r["un_cosines"]) * 0.95)], 4),
            },
        }
    
    with open(output_path, "w") as f:
        json.dump(output_results, f, indent=2)
    
    print(f"\n\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")
    
    # Comparison table
    print(f"\n  Presets per embedder:")
    print(f"  {'Embedder':<35} | dedup_hi | dedup_lo | field  | save   | dim")
    print(f"  {'-'*35}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for name, r in output_results.items():
        p = r["presets"]
        print(f"  {name:<35} | {p['dedup_hi']:<8} | {p['dedup_lo']:<8} | {p['field_minsim']:<8} | {p['save_bind']:<8} | {r['dim']}")
    
    print(f"\n  Separation analysis:")
    print(f"  {'Embedder':<35} | True/Mean | NM/Mean | UN/Mean | True-NM | True-UN | NM-UN")
    print(f"  {'-'*35}-+-{'-'*9}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*5}")
    for name, r in output_results.items():
        s = r["separation"]
        print(f"  {name:<35} | {s['true_mean']:<9} | {s['nm_mean']:<7} | {s['un_mean']:<7} | {s['true_vs_nm_gap']:<7} | {s['true_vs_un_gap']:<7} | {s['nm_vs_un_gap']:<5}")
    
    print(f"\n  Results saved to: {output_path}")
