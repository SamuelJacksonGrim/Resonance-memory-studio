#!/usr/bin/env python3
"""
Round 2 — self-challenge. Uses cached embeddings from embedder_fair_run.py.
Fills haystack for the modes round 1 skipped, then tests:
  - query-conditioned entity filter at retrieval
  - field kNN leak at 0.55 vs scan-best threshold
  - RRF on the haystack store
  - nomic+jina ensemble
  - query-document margin on Naima / hard probes
  - weighted reinforce ON the haystack store
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Reuse helpers
sys.path.insert(0, str(Path(__file__).resolve().parent))
from embedder_fair_run import (
    CACHE, ROOT, load_corpus, generate_haystack, cosine, rf, summarize, auroc,
    tokenize, buckets_of, CONFLICT_PAIRS, proper_nouns, entity_conflict,
    BM25, encode_st, get_or_encode, cache_path, FAMILY, WORK, FRIEND,
    load_models, PROBES,
)

OUT = ROOT / "fire-together-embedder-fair-round2.json"

HAY_TAGS = [
    "nomic__plain", "nomic__search_doc", "nomic__asymmetric",
    "qwen__plain", "qwen__default_query", "qwen__cluster_both",
    "jina__document_both", "jina__asymmetric",
]


def pair_key(a, b):
    return (a, b) if a < b else (b, a)


def retrieval(records, rec_embs, queries, q_embs, k=5, score_fn=None):
    rec_ids = [r["id"] for r in records]
    nm_ids = {r["id"] for r in records if r.get("role") == "near-miss"}
    S = q_embs @ rec_embs.T
    nq = len(queries)
    h1 = 0
    nm_in = 0
    cofire = defaultdict(int)
    theme_nm = defaultdict(lambda: {"n": 0, "hit": 0})
    for qi, q in enumerate(queries):
        if score_fn is None:
            sims = S[qi]
        else:
            sims = np.array([score_fn(q, records[j], float(S[qi, j])) for j in range(len(records))], dtype=np.float64)
        order = np.argsort(-sims)[:k]
        top_ids = [rec_ids[j] for j in order]
        target = q.get("target_id")
        hit = False
        if target:
            hit = target in top_ids
        elif q.get("theme_id"):
            members = [r["id"] for r in records if r.get("theme_id") == q["theme_id"] and r.get("role") == "member"]
            hit = any(i in top_ids for i in members)
        if hit:
            h1 += 1
        leaked = [i for i in top_ids if i in nm_ids]
        if leaked:
            nm_in += 1
        th = q.get("theme_id")
        if th:
            theme_nm[th]["n"] += 1
            nm_id = next((r["id"] for r in records if r.get("theme_id") == th and r.get("role") == "near-miss"), None)
            if nm_id and nm_id in top_ids:
                theme_nm[th]["hit"] += 1
        for i in range(len(top_ids)):
            for j in range(i + 1, len(top_ids)):
                cofire[pair_key(top_ids[i], top_ids[j])] += 1
    return {
        "h1": h1 / nq, "nm@5": nm_in / nq, "cofire": cofire, "nq": nq, "S": S,
        "theme_nm": {t: {"n": v["n"], "hit": v["hit"], "rate": v["hit"] / v["n"] if v["n"] else 0} for t, v in theme_nm.items()},
    }


def cofire_rates(cofire, pairs, nq):
    vals = []
    for p in pairs:
        vals.append(cofire.get(pair_key(p["a"], p["b"]), 0) / nq)
    return float(np.mean(vals)) if vals else 0.0


def pack_ret(ret, corpus):
    t = cofire_rates(ret["cofire"], corpus["true_pairs"], ret["nq"])
    n = cofire_rates(ret["cofire"], corpus["near_miss_pairs"], ret["nq"])
    u = cofire_rates(ret["cofire"], corpus["unrelated_pairs"], ret["nq"])
    return {
        "h1": rf(ret["h1"]), "nm@5": rf(ret["nm@5"]),
        "true_cf": rf(t), "nm_cf": rf(n), "un_cf": rf(u),
        "ratio": rf(t / n if n else None),
        "theme_nm": {k: {**v, "rate": rf(v["rate"])} for k, v in ret["theme_nm"].items()},
    }


def q_conflict_penalty(query_text, doc_text):
    conf = entity_conflict(query_text, doc_text)
    if conf["name_and_conflict"]:
        return 0.30
    if conf["conflict"]:
        return 0.15
    return 0.0


def field_knn(records, rec_embs, min_sim, k=3):
    recs = records
    n = len(recs)
    S = rec_embs @ rec_embs.T
    np.fill_diagonal(S, -2)
    nm_ids = {r["id"] for r in recs if r.get("role") == "near-miss"}
    member_ids = {r["id"] for r in recs if r.get("role") == "member"}
    true_kept = 0
    true_total = 0
    nm_leaked = 0
    nm_total = 0
    un_leaked = 0
    edges = []
    id_of = [r["id"] for r in recs]
    theme_of = [r.get("theme_id") for r in recs]
    role_of = [r.get("role") for r in recs]
    for i, r in enumerate(recs):
        if r.get("role") != "member":
            continue
        order = np.argsort(-S[i])
        kept = []
        for j in order:
            if S[i, j] < min_sim:
                break
            kept.append(j)
            if len(kept) >= k:
                break
        for j in kept:
            edges.append((id_of[i], id_of[j], float(S[i, j]), role_of[j], theme_of[j]))
            if theme_of[j] == theme_of[i] and role_of[j] == "member":
                true_kept += 1
            if role_of[j] == "near-miss":
                nm_leaked += 1
            if theme_of[j] != theme_of[i] and role_of[j] != "near-miss":
                un_leaked += 1
        # how many true neighbors exist
        true_total += sum(1 for j, o in enumerate(recs) if o.get("theme_id") == r["theme_id"] and o.get("role") == "member" and o["id"] != r["id"])
        nm_total += 1  # each member has one theme near-miss
    return {
        "min_sim": min_sim, "k": k,
        "true_edges": true_kept, "true_possible": true_total,
        "true_recall": rf(true_kept / true_total if true_total else None),
        "nm_leaks": nm_leaked, "nm_rate": rf(nm_leaked / nm_total if nm_total else None),
        "un_leaks": un_leaked,
        "edges": [{"a": a, "b": b, "cos": rf(c), "role": role, "theme": th} for a, b, c, role, th in edges],
    }


def rrf_hay(records, rec_embs, queries, q_embs, k=5, rrf_k=60):
    rec_ids = [r["id"] for r in records]
    bm25 = BM25(records)
    S = q_embs @ rec_embs.T
    nq = len(queries)
    h1 = 0
    nm_ids = {r["id"] for r in records if r.get("role") == "near-miss"}
    nm_in = 0
    cofire = defaultdict(int)
    for qi, q in enumerate(queries):
        dense_order = list(np.argsort(-S[qi]))
        sparse = bm25.score_all(q["query"])
        scores = {}
        for rank, idx in enumerate(dense_order):
            scores[rec_ids[idx]] = scores.get(rec_ids[idx], 0.0) + 1.0 / (rrf_k + rank + 1)
        for rnk, (did, _) in enumerate(sparse):
            scores[did] = scores.get(did, 0.0) + 1.0 / (rrf_k + rnk + 1)
        top = sorted(scores.items(), key=lambda x: -x[1])[:k]
        top_ids = [t[0] for t in top]
        target = q.get("target_id")
        hit = False
        if target:
            hit = target in top_ids
        elif q.get("theme_id"):
            members = [r["id"] for r in records if r.get("theme_id") == q["theme_id"] and r.get("role") == "member"]
            hit = any(i in top_ids for i in members)
        if hit:
            h1 += 1
        if any(i in nm_ids for i in top_ids):
            nm_in += 1
        for i in range(len(top_ids)):
            for j in range(i + 1, len(top_ids)):
                cofire[pair_key(top_ids[i], top_ids[j])] += 1
    return pack_ret({"h1": h1 / nq, "nm@5": nm_in / nq, "cofire": cofire, "nq": nq, "theme_nm": {}}, {
        "true_pairs": [],  # filled by caller
    })  # placeholder — we'll compute below with corpus


def reinforce_hay(records, rec_embs, queries, q_embs, k=5, sessions=10, weighted=False, field_minsim=0.55):
    rec_ids = [r["id"] for r in records]
    rec_by = {r["id"]: i for i, r in enumerate(records)}
    S = q_embs @ rec_embs.T
    themes = sorted({r.get("theme_id") for r in records if r.get("theme_id")})
    weights = defaultdict(float)
    ALPHA = 0.1
    MAX_BONUS = 0.3
    q_by = defaultdict(list)
    for i, q in enumerate(queries):
        if q.get("kind") == "member-paraphrase" and q.get("theme_id"):
            q_by[q["theme_id"]].append(i)

    def bump(ids, sims_by_id):
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                if a not in rec_by or b not in rec_by:
                    continue
                alpha = ALPHA
                if weighted:
                    ea, eb = rec_embs[rec_by[a]], rec_embs[rec_by[b]]
                    dd = max(0.0, cosine(ea, eb))
                    gate = max(0.0, (dd - field_minsim) / max(1e-6, 1.0 - field_minsim))
                    sa = max(0.0, sims_by_id.get(a, 0.0))
                    sb = max(0.0, sims_by_id.get(b, 0.0))
                    alpha = ALPHA * gate * math.sqrt(sa * sb)
                weights[pair_key(a, b)] += alpha

    last = None
    for sess in range(sessions):
        for th in themes:
            for qi in q_by.get(th, []):
                sims = S[qi]
                order = np.argsort(-sims)[:k]
                top_ids = [rec_ids[j] for j in order]
                sims_by = {rec_ids[j]: float(sims[j]) for j in order}
                bump(top_ids, sims_by)

        def mean_w(pairs):
            return float(np.mean([weights[pair_key(p["a"], p["b"])] for p in pairs])) if pairs else 0.0

        def mean_norm(pairs):
            node_max = defaultdict(float)
            for (a, b), w in weights.items():
                node_max[a] = max(node_max[a], w)
                node_max[b] = max(node_max[b], w)
            vals = []
            for p in pairs:
                w = weights[pair_key(p["a"], p["b"])]
                mx = max(node_max[p["a"]], node_max[p["b"]], 1e-9)
                vals.append(MAX_BONUS * math.tanh(w / mx))
            return float(np.mean(vals)) if vals else 0.0

        last = {
            "session": sess,
            "w_true": mean_w,  # filled below
        }
    return weights


def qdoc_margins(records, rec_embs, queries, q_embs):
    """For each theme, on that theme's member-paraphrase queries, mean sim to
    members vs mean sim to the theme near-miss."""
    rec_ids = [r["id"] for r in records]
    id2i = {r["id"]: i for i, r in enumerate(records)}
    S = q_embs @ rec_embs.T
    out = {}
    for th in sorted({r.get("theme_id") for r in records if r.get("theme_id")}):
        members = [r["id"] for r in records if r.get("theme_id") == th and r.get("role") == "member"]
        nm = next((r["id"] for r in records if r.get("theme_id") == th and r.get("role") == "near-miss"), None)
        if not nm:
            continue
        m_sims, n_sims, margins = [], [], []
        n_rank = []
        for qi, q in enumerate(queries):
            if q.get("kind") != "member-paraphrase" or q.get("theme_id") != th:
                continue
            ms = [float(S[qi, id2i[m]]) for m in members]
            ns = float(S[qi, id2i[nm]])
            m_sims.append(float(np.mean(ms)))
            n_sims.append(ns)
            margins.append(float(np.mean(ms)) - ns)
            # rank of near-miss (1-based)
            order = np.argsort(-S[qi])
            ranks = {rec_ids[j]: rnk + 1 for rnk, j in enumerate(order)}
            n_rank.append(ranks[nm])
        out[th] = {
            "q_to_member_mean": rf(float(np.mean(m_sims))),
            "q_to_nm_mean": rf(float(np.mean(n_sims))),
            "margin": rf(float(np.mean(margins))),
            "nm_mean_rank": rf(float(np.mean(n_rank))),
            "nm_in_top5": rf(float(np.mean([r <= 5 for r in n_rank]))),
        }
    return out


def main():
    corpus = load_corpus()
    hay = generate_haystack(300, seed=42)
    rec_texts = [r["text"] for r in corpus["records"]]
    q_texts = [q["query"] for q in corpus["queries"]]
    hay_texts = [h["text"] for h in hay]
    all_recs = corpus["records"] + hay

    # Load cached embeddings; encode missing haystack.
    # We don't reload models unless a haystack cache is missing.
    need_model = []
    for tag in HAY_TAGS:
        if not cache_path(tag, "haystack").exists() or not cache_path(tag, "records").exists():
            need_model.append(tag)
    models = {}
    if need_model:
        print("need models for", need_model, flush=True)
        models = load_models()

    from embedder_fair_run import mode_specs
    specs = {s[0]: s for s in mode_specs(models)} if models else {}

    embs = {}
    for tag in HAY_TAGS:
        rec_e = np.load(cache_path(tag, "records"))
        q_e = np.load(cache_path(tag, "queries"))
        hay_p = cache_path(tag, "haystack")
        if hay_p.exists():
            hay_e = np.load(hay_p)
            print("cache hay", tag, hay_e.shape, flush=True)
        else:
            print("encode hay", tag, flush=True)
            _, _, doc_fn, _, _, _, _ = specs[tag]
            hay_e = get_or_encode(tag, "haystack", hay_texts, doc_fn)
        embs[tag] = {"rec": rec_e, "q": q_e, "hay": hay_e, "all": np.vstack([rec_e, hay_e])}

    results = {"haystack": {}, "qfilter": {}, "field_knn": {}, "rrf_hay": {}, "ensemble": {}, "qdoc": {}, "reinforce_hay": {}, "notes": []}

    for tag in HAY_TAGS:
        print("hay retrieval", tag, flush=True)
        ret = retrieval(all_recs, embs[tag]["all"], corpus["queries"], embs[tag]["q"], k=5)
        results["haystack"][tag] = pack_ret(ret, corpus)
        results["qdoc"][tag] = qdoc_margins(all_recs, embs[tag]["all"], corpus["queries"], embs[tag]["q"])

        def sfn(q, rec, sim, _tag=tag):
            return sim - q_conflict_penalty(q["query"], rec["text"])
        ret_f = retrieval(all_recs, embs[tag]["all"], corpus["queries"], embs[tag]["q"], k=5, score_fn=sfn)
        results["qfilter"][tag] = pack_ret(ret_f, corpus)

        # field kNN on the 25 core records (Related: is over stored memories)
        results["field_knn"][tag] = {
            "at_0.55": field_knn(corpus["records"], embs[tag]["rec"], 0.55, k=3),
            "at_0.65": field_knn(corpus["records"], embs[tag]["rec"], 0.65, k=3),
            "at_0.70": field_knn(corpus["records"], embs[tag]["rec"], 0.70, k=3),
        }
        # drop bulky edge lists from 0.55/0.65 keep 0.70 edges for inspection
        for t in ("at_0.55", "at_0.65"):
            results["field_knn"][tag][t] = {k: v for k, v in results["field_knn"][tag][t].items() if k != "edges"}
        results["field_knn"][tag]["at_0.70"]["nm_edges"] = [
            e for e in results["field_knn"][tag]["at_0.70"]["edges"] if e["role"] == "near-miss"
        ]
        results["field_knn"][tag]["at_0.70"] = {
            k: v for k, v in results["field_knn"][tag]["at_0.70"].items() if k != "edges"
        }

        # RRF on haystack
        rec_ids = [r["id"] for r in all_recs]
        bm25 = BM25(all_recs)
        S = embs[tag]["q"] @ embs[tag]["all"].T
        nq = len(corpus["queries"])
        h1 = 0
        nm_ids = {r["id"] for r in all_recs if r.get("role") == "near-miss"}
        nm_in = 0
        cofire = defaultdict(int)
        rrf_k = 60
        k = 5
        for qi, q in enumerate(corpus["queries"]):
            dense_order = list(np.argsort(-S[qi]))
            sparse = bm25.score_all(q["query"])
            scores = {}
            for rank, idx in enumerate(dense_order):
                scores[rec_ids[idx]] = scores.get(rec_ids[idx], 0.0) + 1.0 / (rrf_k + rank + 1)
            for rnk, (did, _) in enumerate(sparse):
                scores[did] = scores.get(did, 0.0) + 1.0 / (rrf_k + rnk + 1)
            top = sorted(scores.items(), key=lambda x: -x[1])[:k]
            top_ids = [t[0] for t in top]
            target = q.get("target_id")
            hit = False
            if target:
                hit = target in top_ids
            elif q.get("theme_id"):
                members = [r["id"] for r in all_recs if r.get("theme_id") == q["theme_id"] and r.get("role") == "member"]
                hit = any(i in top_ids for i in members)
            if hit:
                h1 += 1
            if any(i in nm_ids for i in top_ids):
                nm_in += 1
            for i in range(len(top_ids)):
                for j in range(i + 1, len(top_ids)):
                    cofire[pair_key(top_ids[i], top_ids[j])] += 1
        results["rrf_hay"][tag] = pack_ret({"h1": h1 / nq, "nm@5": nm_in / nq, "cofire": cofire, "nq": nq, "theme_nm": {}}, corpus)

    # Ensemble: mean of nomic__plain and jina__document_both doc-doc scores (same query emb? no —
    # different spaces. Ensemble RANK fusion via RRF of the two dense arms, plus optional BM25.
    print("ensemble nomic_plain + jina_document RRF on haystack", flush=True)
    rec_ids = [r["id"] for r in all_recs]
    S_n = embs["nomic__plain"]["q"] @ embs["nomic__plain"]["all"].T
    S_j = embs["jina__document_both"]["q"] @ embs["jina__document_both"]["all"].T
    bm25 = BM25(all_recs)
    nq = len(corpus["queries"])
    for name, arms in (
        ("nomic+jina", [S_n, S_j]),
        ("nomic+jina+bm25", [S_n, S_j, None]),
    ):
        h1 = 0
        nm_ids = {r["id"] for r in all_recs if r.get("role") == "near-miss"}
        nm_in = 0
        cofire = defaultdict(int)
        k = 5
        rrf_k = 60
        for qi, q in enumerate(corpus["queries"]):
            scores = {}
            for S in arms:
                if S is None:
                    sparse = bm25.score_all(q["query"])
                    for rnk, (did, _) in enumerate(sparse):
                        scores[did] = scores.get(did, 0.0) + 1.0 / (rrf_k + rnk + 1)
                else:
                    order = list(np.argsort(-S[qi]))
                    for rank, idx in enumerate(order):
                        scores[rec_ids[idx]] = scores.get(rec_ids[idx], 0.0) + 1.0 / (rrf_k + rank + 1)
            top = sorted(scores.items(), key=lambda x: -x[1])[:k]
            top_ids = [t[0] for t in top]
            target = q.get("target_id")
            hit = False
            if target:
                hit = target in top_ids
            elif q.get("theme_id"):
                members = [r["id"] for r in all_recs if r.get("theme_id") == q["theme_id"] and r.get("role") == "member"]
                hit = any(i in top_ids for i in members)
            if hit:
                h1 += 1
            if any(i in nm_ids for i in top_ids):
                nm_in += 1
            for i in range(len(top_ids)):
                for j in range(i + 1, len(top_ids)):
                    cofire[pair_key(top_ids[i], top_ids[j])] += 1
        results["ensemble"][name] = pack_ret({"h1": h1 / nq, "nm@5": nm_in / nq, "cofire": cofire, "nq": nq, "theme_nm": {}}, corpus)

    # Weighted reinforce on haystack for nomic__plain (production geometry)
    print("reinforce hay nomic__plain", flush=True)
    for tag, fmin in (("nomic__plain", 0.55), ("nomic__plain_gate070", 0.70), ("jina__document_both", 0.33)):
        src = "nomic__plain" if tag.startswith("nomic") else tag
        rec_ids = [r["id"] for r in all_recs]
        rec_by = {r["id"]: i for i, r in enumerate(all_recs)}
        S = embs[src]["q"] @ embs[src]["all"].T
        rec_embs = embs[src]["all"]
        themes = [t["id"] for t in corpus["themes"]]
        q_by = defaultdict(list)
        for i, q in enumerate(corpus["queries"]):
            if q.get("kind") == "member-paraphrase" and q.get("theme_id"):
                q_by[q["theme_id"]].append(i)
        out_eq = []
        out_w = []
        for weighted, bucket in ((False, out_eq), (True, out_w)):
            weights = defaultdict(float)
            ALPHA = 0.1
            MAX_BONUS = 0.3
            for sess in range(10):
                for th in themes:
                    for qi in q_by.get(th, []):
                        sims = S[qi]
                        order = np.argsort(-sims)[:5]
                        top_ids = [rec_ids[j] for j in order]
                        sims_by = {rec_ids[j]: float(sims[j]) for j in order}
                        for i in range(len(top_ids)):
                            for j in range(i + 1, len(top_ids)):
                                a, b = top_ids[i], top_ids[j]
                                if a not in rec_by or b not in rec_by:
                                    continue
                                alpha = ALPHA
                                if weighted:
                                    ea, eb = rec_embs[rec_by[a]], rec_embs[rec_by[b]]
                                    dd = max(0.0, cosine(ea, eb))
                                    gate = max(0.0, (dd - fmin) / max(1e-6, 1.0 - fmin))
                                    sa = max(0.0, sims_by.get(a, 0.0))
                                    sb = max(0.0, sims_by.get(b, 0.0))
                                    alpha = ALPHA * gate * math.sqrt(max(sa, 0) * max(sb, 0))
                                weights[pair_key(a, b)] += alpha
                def mean_w(pairs):
                    return float(np.mean([weights[pair_key(p["a"], p["b"])] for p in pairs]))
                def mean_bonus(pairs):
                    return float(np.mean([MAX_BONUS * math.tanh(weights[pair_key(p["a"], p["b"])]) for p in pairs]))
                def mean_norm(pairs):
                    node_max = defaultdict(float)
                    for (a, b), w in weights.items():
                        node_max[a] = max(node_max[a], w)
                        node_max[b] = max(node_max[b], w)
                    vals = []
                    for p in pairs:
                        w = weights[pair_key(p["a"], p["b"])]
                        mx = max(node_max[p["a"]], node_max[p["b"]], 1e-9)
                        vals.append(MAX_BONUS * math.tanh(w / mx))
                    return float(np.mean(vals))
                tw, nw, uw = mean_w(corpus["true_pairs"]), mean_w(corpus["near_miss_pairs"]), mean_w(corpus["unrelated_pairs"])
                tn, nn, un = mean_norm(corpus["true_pairs"]), mean_norm(corpus["near_miss_pairs"]), mean_norm(corpus["unrelated_pairs"])
                tb, nb, ub = mean_bonus(corpus["true_pairs"]), mean_bonus(corpus["near_miss_pairs"]), mean_bonus(corpus["unrelated_pairs"])
                bucket.append({
                    "session": sess,
                    "w_true": rf(tw), "w_nm": rf(nw), "w_un": rf(uw), "w_ratio": rf(tw / nw if nw else None),
                    "bonus_ratio": rf(tb / nb if nb else None),
                    "norm_ratio": rf(tn / nn if nn else None),
                    "norm_un": rf(un), "w_un": rf(uw),
                })
        results["reinforce_hay"][tag] = {"equal": out_eq[-1], "weighted": out_w[-1]}

    results["notes"] = [
        "Haystack (300 distractors, seed 42) is the honest cofire test; 25-record nm@5 was saturated.",
        "Query-conditioned entity filter subtracts 0.15/0.30 from a candidate if the QUERY and the DOC have conflicting relation/category buckets.",
        "Field kNN is the Related: gate (doc-doc), independent of cofire. 0.55 is production; 0.70 is scan-best on nomic plain.",
        "Ensemble is RRF of nomic-plain ranks and jina-document ranks (incommensurable cosine scales — ranks only).",
        "Weighted reinforce on the haystack store: alpha scaled by how far doc-doc cosine sits above the gate. Unrelated must stay low.",
    ]

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items() if not str(k).startswith("_")}
        if isinstance(o, list):
            return [clean(x) for x in o]
        if isinstance(o, float):
            return rf(o)
        return o

    OUT.write_text(json.dumps(clean(results), indent=2), encoding="utf-8")
    print("wrote", OUT, flush=True)

    # stdout digest
    print("\n=== HAYSTACK DENSE ===")
    for tag, r in results["haystack"].items():
        print(f"  {tag:24} h1={r['h1']} nm@5={r['nm@5']} ratio={r['ratio']} un={r['un_cf']}")
    print("=== QFILTER ===")
    for tag, r in results["qfilter"].items():
        print(f"  {tag:24} h1={r['h1']} nm@5={r['nm@5']} ratio={r['ratio']} un={r['un_cf']}")
    print("=== FIELD KNN ===")
    for tag, r in results["field_knn"].items():
        for t, v in r.items():
            if t.startswith("at_"):
                print(f"  {tag:24} {t} true_recall={v['true_recall']} nm_leaks={v['nm_leaks']} un_leaks={v['un_leaks']}")
    print("=== RRF HAY ===")
    for tag, r in results["rrf_hay"].items():
        print(f"  {tag:24} h1={r['h1']} nm@5={r['nm@5']} ratio={r['ratio']} un={r['un_cf']}")
    print("=== ENSEMBLE ===")
    for tag, r in results["ensemble"].items():
        print(f"  {tag:24} h1={r['h1']} nm@5={r['nm@5']} ratio={r['ratio']} un={r['un_cf']}")
    print("=== QDOC MARGINS ===")
    for tag, r in results["qdoc"].items():
        print(tag)
        for th, v in r.items():
            print(f"  {th:16} member={v['q_to_member_mean']} nm={v['q_to_nm_mean']} margin={v['margin']} nm_rank={v['nm_mean_rank']} in5={v['nm_in_top5']}")
    print("=== REINFORCE HAY ===")
    for tag, r in results["reinforce_hay"].items():
        print(tag, "equal", r["equal"])
        print(tag, "weighted", r["weighted"])


if __name__ == "__main__":
    main()
