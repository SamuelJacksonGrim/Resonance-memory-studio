#!/usr/bin/env python3
"""Ablate query-filter: relation-only vs full theme-fitted buckets."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from embedder_fair_run import (
    load_corpus, generate_haystack, entity_conflict, FAMILY, WORK, FRIEND,
    buckets_of, CONFLICT_PAIRS, cache_path, tokenize, rf,
)

corpus = load_corpus()
hay = generate_haystack(300, seed=42)
all_recs = corpus["records"] + hay
queries = corpus["queries"]

REL_ONLY = {"family": FAMILY, "work": WORK, "friend": FRIEND}
REL_CONFLICTS = {
    frozenset({"family", "work"}),
    frozenset({"family", "friend"}),
    frozenset({"work", "friend"}),
}

def buckets(text, table):
    toks = set(tokenize(text))
    return {n for n, v in table.items() if toks & v}

def rel_conflict(q, d):
    bq, bd = buckets(q, REL_ONLY), buckets(d, REL_ONLY)
    hit = False
    for pair in REL_CONFLICTS:
        x, y = tuple(pair)
        if (x in bq and y in bd) or (y in bq and x in bd):
            hit = True
    names_q = {n.lower() for n in __import__("embedder_fair_run", fromlist=["proper_nouns"]).proper_nouns(q)}
    names_d = {n.lower() for n in __import__("embedder_fair_run", fromlist=["proper_nouns"]).proper_nouns(d)}
    shared = names_q & names_d
    if hit and shared:
        return 0.30
    if hit:
        return 0.15
    return 0.0

def full_pen(q, d):
    conf = entity_conflict(q, d)
    if conf["name_and_conflict"]:
        return 0.30
    if conf["conflict"]:
        return 0.15
    return 0.0

def pair_key(a, b):
    return (a, b) if a < b else (b, a)

def run(tag, penalty_fn):
    rec = np.load(cache_path(tag, "records"))
    qe = np.load(cache_path(tag, "queries"))
    haye = np.load(cache_path(tag, "haystack"))
    all_e = np.vstack([rec, haye])
    rec_ids = [r["id"] for r in all_recs]
    nm_ids = {r["id"] for r in all_recs if r.get("role") == "near-miss"}
    S = qe @ all_e.T
    nq = len(queries)
    h1 = 0
    nm_in = 0
    cofire = defaultdict(int)
    theme_nm = defaultdict(lambda: [0, 0])
    h1_miss = []
    for qi, q in enumerate(queries):
        sims = np.array([float(S[qi, j]) - penalty_fn(q["query"], all_recs[j]["text"]) for j in range(len(all_recs))])
        order = np.argsort(-sims)[:5]
        top_ids = [rec_ids[j] for j in order]
        target = q.get("target_id")
        hit = False
        if target:
            hit = target in top_ids
        elif q.get("theme_id"):
            members = [r["id"] for r in all_recs if r.get("theme_id") == q["theme_id"] and r.get("role") == "member"]
            hit = any(i in top_ids for i in members)
        if hit:
            h1 += 1
        else:
            h1_miss.append((q.get("kind"), q.get("theme_id"), q["query"][:70], top_ids[:5]))
        if any(i in nm_ids for i in top_ids):
            nm_in += 1
        th = q.get("theme_id")
        if th:
            theme_nm[th][0] += 1
            nm_id = next((r["id"] for r in all_recs if r.get("theme_id") == th and r.get("role") == "near-miss"), None)
            if nm_id in top_ids:
                theme_nm[th][1] += 1
        for i in range(len(top_ids)):
            for j in range(i + 1, len(top_ids)):
                cofire[pair_key(top_ids[i], top_ids[j])] += 1
    def mean_rate(pairs):
        return float(np.mean([cofire.get(pair_key(p["a"], p["b"]), 0) / nq for p in pairs]))
    t, n, u = mean_rate(corpus["true_pairs"]), mean_rate(corpus["near_miss_pairs"]), mean_rate(corpus["unrelated_pairs"])
    return {
        "h1": h1 / nq, "nm@5": nm_in / nq, "ratio": t / n if n else None,
        "true_cf": t, "nm_cf": n, "un_cf": u,
        "theme": {th: {"n": a, "hit": b, "rate": b / a} for th, (a, b) in theme_nm.items()},
        "h1_miss": h1_miss[:8], "n_miss": len(h1_miss),
    }

tag = "nomic__plain"
none = run(tag, lambda q, d: 0.0)
rel = run(tag, rel_conflict)
full = run(tag, full_pen)
out = {"none": none, "relation_only": rel, "full_lexicon": full}
# strip long miss lists for json except counts
for k in out:
    out[k]["h1_miss_sample"] = out[k].pop("h1_miss")
Path("eval/substrate/_ablate_qfilter.json").write_text(json.dumps(out, indent=2, default=rf), encoding="utf-8")
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "h1_miss_sample"} for k, v in out.items()}, indent=2))
print("H1 misses relation_only", out["relation_only"]["n_miss"], out["relation_only"]["h1_miss_sample"])
print("H1 misses full", out["full_lexicon"]["n_miss"], out["full_lexicon"]["h1_miss_sample"])
