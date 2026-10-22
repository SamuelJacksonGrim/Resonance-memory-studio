#!/usr/bin/env python3
"""
Fair embedder comparison + hybrid / entity / adapter exploration.

Prior run (fire-together-embedder-comparison.py) encoded every model PLAIN.
That measured format-tolerance, not retrieval geometry. This run invokes each
model the way its authors specify, derives presets from THAT geometry, then
pushes past off-the-shelf cosine: lexical hybrid, entity-conflict, name-mask
ablation, leave-one-theme-out linear adapter, similarity-weighted reinforce.

Measured, never asserted. Unrelated pairs are a safety check on every scheme.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
CORPUS_LITE = ROOT / "_corpus-lite.json"
CORPUS_FULL = REPO / "eval" / "corpora" / "fire-together-corpus.json"
DUPLICATES = REPO / "eval" / "corpora" / "duplicates.jsonl"
CACHE = ROOT / ".cache-embedder-fair"
OUT_JSON = ROOT / "fire-together-embedder-fair.json"
OUT_MD = ROOT / "fire-together-embedder-fair-report.md"

NOMIC_PATH = Path(r"C:\Users\spamw\.cache\huggingface\hub\models--nomic-ai--nomic-embed-text-v1.5\snapshots\e9b6763023c676ca8431644204f50c2b100d9aab")
QWEN_PATH = Path(r"C:\Users\spamw\.cache\huggingface\hub\models--Qwen--Qwen3-Embedding-0.6B\snapshots\97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3")
JINA_PATH = Path(r"C:\Users\spamw\.cache\huggingface\hub\models--jinaai--jina-embeddings-v5-text-nano-retrieval\snapshots\ac5d898c8d382b17167c33e5c8af644a3519b47d")

# Qwen default retrieval instruction (from the model's own prompts map).
QWEN_QUERY_DEFAULT = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
)
QWEN_STS = "Instruct: Retrieve semantically similar text.\nQuery:"
QWEN_CLUSTER = (
    "Instruct: Represent this personal fact for clustering related memories; "
    "distinguish different people, places, and objects even if they share a name or category.\nQuery:"
)
QWEN_MEMORY = (
    "Instruct: Given a question about someone's personal memories, retrieve the stored fact "
    "that answers it. Treat different people as distinct even when they share a name; treat "
    "different objects as distinct even when they share a category.\nQuery:"
)

STOPWORDS = {
    "a", "an", "the", "i", "i'm", "im", "i've", "ive", "my", "me", "mine", "we", "our",
    "you", "your", "is", "am", "are", "was", "were", "be", "been", "being", "to", "of",
    "in", "on", "at", "for", "with", "and", "or", "but", "from", "as", "by", "it", "its",
    "this", "that", "these", "those", "do", "does", "did", "have", "has", "had", "not",
    "no", "so", "if", "then", "than", "too", "very", "just", "about", "into", "over",
    "every", "right", "before", "after", "where", "when", "what", "who", "how", "which",
}

FAMILY = {
    "sister", "brother", "mother", "father", "mom", "dad", "wife", "husband",
    "son", "daughter", "cousin", "uncle", "aunt", "sibling", "parent", "child",
}
WORK = {
    "coworker", "colleague", "boss", "manager", "teammate", "employee",
    "coworker", "co-worker", "infrastructure", "frontend", "engineer",
}
FRIEND = {"friend", "roommate", "neighbor", "buddy"}
COFFEE = {"cortado", "coffee", "espresso", "oat", "oat-milk", "oatmilk"}
TEA = {"tea", "chamomile", "honey"}
CLIMB = {"bouldering", "climbing", "climb", "gym"}
LIFT = {"weights", "weightlifting", "fitness", "lift", "downtown"}
SHELF = {"bookshelf", "bookcase", "walnut", "shelves", "handmade", "workshop"}
DESK = {"desk", "ikea", "pine", "flat-pack", "flatpack", "study"}
ALLERGY = {"allergy", "allergic", "anaphylactic", "penicillin", "amoxicillin", "bracelet"}
SUPPLEMENT = {"vitamin", "supplement", "supplements", "zinc", "breakfast"}

BUCKETS = {
    "family": FAMILY,
    "work": WORK,
    "friend": FRIEND,
    "coffee": COFFEE,
    "tea": TEA,
    "climb": CLIMB,
    "lift": LIFT,
    "shelf": SHELF,
    "desk": DESK,
    "allergy": ALLERGY,
    "supplement": SUPPLEMENT,
}
CONFLICT_PAIRS = {
    frozenset({"family", "work"}),
    frozenset({"family", "friend"}),
    frozenset({"work", "friend"}),
    frozenset({"coffee", "tea"}),
    frozenset({"climb", "lift"}),
    frozenset({"shelf", "desk"}),
    frozenset({"allergy", "supplement"}),
}

# Extra probes that isolate confound classes the 5-theme set only samples once.
PROBES = [
    # A: shared rare name + conflicting relation (Naima-class)
    {"id": "A1", "cls": "shared-name-conflict-relation",
     "a": "My sister Naima teaches chemistry at the local high school",
     "b": "My coworker Naima is a frontend engineer on the infrastructure team",
     "expect": "near-miss"},
    {"id": "A2", "cls": "shared-name-conflict-relation",
     "a": "My brother Omar is a dentist in Portland",
     "b": "My neighbor Omar is a dentist in Portland",
     "expect": "near-miss"},
    {"id": "A3", "cls": "shared-name-conflict-relation",
     "a": "My wife Priya is a painter who shows at the downtown gallery",
     "b": "My colleague Priya is a painter who shows at the downtown gallery",
     "expect": "near-miss"},
    {"id": "A4", "cls": "shared-name-conflict-relation",
     "a": "I call my sister Naima every Sunday evening to catch up",
     "b": "I call my coworker Naima every Sunday evening to catch up",
     "expect": "near-miss"},
    # A': same name, SAME relation (should stay high — same person restatement)
    {"id": "Ap1", "cls": "shared-name-same-relation",
     "a": "My sister Naima teaches chemistry at the local high school",
     "b": "Naima is my sister and she works as a high school chemistry teacher",
     "expect": "true"},
    {"id": "Ap2", "cls": "shared-name-same-relation",
     "a": "My brother Omar is a dentist in Portland",
     "b": "Omar, my brother, runs a dental practice in Portland",
     "expect": "true"},
    # B: same relation+role, different name (two sisters — related-category, not same person)
    {"id": "B1", "cls": "same-relation-different-name",
     "a": "My sister Naima teaches chemistry at the local high school",
     "b": "My sister Layla teaches chemistry at the local high school",
     "expect": "near-miss"},
    {"id": "B2", "cls": "same-relation-different-name",
     "a": "My coworker Naima is a frontend engineer",
     "b": "My coworker Jordan is a frontend engineer",
     "expect": "near-miss"},
    # C: shared category, different instance (cortado-class)
    {"id": "C1", "cls": "shared-category",
     "a": "I drink an oat-milk cortado every morning",
     "b": "I drink hot chamomile tea with honey at night before bed",
     "expect": "near-miss"},
    {"id": "C2", "cls": "shared-category",
     "a": "I am building a custom walnut bookshelf in my garage workshop",
     "b": "I bought a flat-pack pine desk from IKEA for my study",
     "expect": "near-miss"},
    {"id": "C3", "cls": "shared-category",
     "a": "I climb at the south bouldering gym on Tuesday evenings",
     "b": "I lift free weights at the downtown fitness center on Thursday mornings",
     "expect": "near-miss"},
    {"id": "C4", "cls": "shared-category",
     "a": "I have a severe allergic reaction to penicillin and amoxicillin",
     "b": "I take daily vitamin D and zinc supplements with breakfast",
     "expect": "near-miss"},
    # D: restatements (must stay high)
    {"id": "D1", "cls": "restatement",
     "a": "I'm allergic to penicillin",
     "b": "I am allergic to penicillin",
     "expect": "true"},
    {"id": "D2", "cls": "restatement",
     "a": "I prefer tea over coffee",
     "b": "I like tea more than coffee",
     "expect": "true"},
    {"id": "D3", "cls": "restatement",
     "a": "I climb at the south bouldering gym on Tuesday evenings",
     "b": "Tuesday nights I go indoor bouldering at the south gym",
     "expect": "true"},
    # E: truly unrelated
    {"id": "E1", "cls": "unrelated",
     "a": "I drink an oat-milk cortado every morning",
     "b": "The capital of Oregon is Salem, which was incorporated in 1857",
     "expect": "unrelated"},
    {"id": "E2", "cls": "unrelated",
     "a": "My sister Naima teaches chemistry at the local high school",
     "b": "I replaced the furnace filter with an allergen-rated pleated filter",
     "expect": "unrelated"},
    {"id": "E3", "cls": "unrelated",
     "a": "I am building a custom walnut bookshelf in my garage workshop",
     "b": "Apollo 11 landed on the lunar surface on July 20, 1969",
     "expect": "unrelated"},
    # F: structure-only (no shared rare name) — sister-chemist vs coworker-engineer
    {"id": "F1", "cls": "structure-only",
     "a": "My sister teaches chemistry at the local high school",
     "b": "My coworker is a frontend engineer on the infrastructure team",
     "expect": "near-miss"},
    {"id": "F2", "cls": "structure-only",
     "a": "My sister teaches chemistry at the local high school",
     "b": "My sister works as a high school chemistry teacher",
     "expect": "true"},
]


def log(msg):
    print(msg, flush=True)


def load_corpus():
    src = CORPUS_LITE if CORPUS_LITE.exists() else CORPUS_FULL
    raw = json.loads(src.read_text(encoding="utf-8"))
    records = [{"id": r["id"], "text": r["text"], "theme_id": r.get("theme_id"), "role": r.get("role")}
               for r in raw["records"]]
    queries = [{"query": q["query"], "kind": q.get("kind"), "theme_id": q.get("theme_id"),
                "target_id": q.get("target_id")} for q in raw["queries"]]
    return {
        "themes": raw["themes"],
        "records": records,
        "queries": queries,
        "true_pairs": raw["true_pairs"],
        "near_miss_pairs": raw["near_miss_pairs"],
        "unrelated_pairs": raw["unrelated_pairs"],
    }


def load_duplicates():
    groups = defaultdict(list)
    controls = []
    with DUPLICATES.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("role") != "write":
                continue
            item = {"text": rec["text"], "band": rec.get("band"), "group": rec.get("dup_group")}
            if rec.get("band") == "control":
                controls.append(item)
            else:
                groups[rec["dup_group"]].append(item)
    return groups, controls


def generate_haystack(n=300, seed=42):
    rng = random.Random(seed)
    pick = rng.choice
    templates = [
        lambda: f"I bought a new {pick(['blue','red','green','black','white','gray','navy','brown'])} {pick(['jacket','shirt','pants','dress','sweater','coat','hoodie','scarf','hat','belt'])} from {pick(['Zara','H&M','Uniqlo','Target','Walmart','Amazon','Nordstrom','Old Navy','Gap','Costco'])} last week",
        lambda: f"I've been cooking {pick(['Thai','Italian','Mexican','Japanese','Indian','Korean','French','Chinese','Vietnamese','Lebanese'])} food at home more often lately",
        lambda: f"My {pick(['dad','mom','brother','cousin','uncle','aunt','neighbor','friend','colleague'])} lives in {pick(['Denver','Austin','Portland','Seattle','Chicago','Boston','Miami','Atlanta','Nashville','Minneapolis'])}",
        lambda: f"I need to {pick(['update my resume','schedule a dentist appointment','renew my passport','pay the electric bill','change my oil','fix the leaky faucet','clean the gutters','file my taxes'])}",
        lambda: f"I read a great book about {pick(['neuroscience','history','cooking','travel','philosophy','psychology','astronomy','genetics','art','politics'])} this past weekend",
        lambda: f"The {pick(['parking garage','grocery store','coffee shop','library','hospital','bank','post office','hardware store','pharmacy'])} on {pick(['Main','Oak','Elm','Maple','Cedar','Pine','Birch','Chestnut','Willow'])} Street is always crowded",
        lambda: f"I'm trying to cut back on {pick(['sugar','caffeine','social media','screen time','carbs','dairy','processed food','meat','alcohol','junk food'])}",
        lambda: f"We had {pick(['pizza','sushi','tacos','ramen','burgers','pasta','curry','steak','bbq','falafel'])} for dinner last night",
        lambda: f"I went {pick(['hiking','swimming','biking','running','kayaking','surfing','fishing','camping','skiing'])} last weekend",
        lambda: f"I've been listening to a lot of {pick(['podcasts','lo-fi','jazz','classical','rock','hip-hop','indie','folk','electronic','country'])}",
        lambda: f"I should really {pick(['exercise more','eat better','sleep more','save money','travel more','learn a new skill','call my mom','clean the house','start meditating'])}",
        lambda: f"I bought a {pick(['laptop','phone','tablet','monitor','keyboard','headphones','camera','watch','speaker'])} from {pick(['Apple','Samsung','Dell','Sony','Bose','Canon','Garmin','JBL','Anker'])}",
        lambda: f"I went to {pick(['the beach','the mountains','the lake','the desert','the forest','the canyon','the waterfall','the island','the river'])} and the {pick(['view','weather','water','air','scenery','sunrise','sunset','stars'])} was incredible",
        lambda: f"The {pick(['traffic','construction','road work','accident','detour','lane closure','bridge repair','street fair','parade'])} made me {pick(['late','take a detour','wait an hour','find an alternate route','lose my patience','replan'])}",
        lambda: f"I renewed my {pick(['library card','gym membership','car insurance','driver license','magazine subscription','domain name'])} last month",
    ]
    recs = []
    for i in range(n):
        recs.append({"id": f"h{i}", "text": templates[i % len(templates)](), "theme_id": None, "role": "haystack"})
    return recs


def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", str(text).lower())


def proper_nouns(text):
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]*", str(text))
    out = []
    for i, w in enumerate(words):
        if i == 0:
            continue
        if w[0].isupper() and w.lower() not in STOPWORDS:
            out.append(w)
    return out


def mask_proper_nouns(text, placeholder="NAME"):
    names = proper_nouns(text)
    out = text
    for n in sorted(set(names), key=len, reverse=True):
        out = re.sub(rf"\b{re.escape(n)}\b", placeholder, out)
    return out


def mask_shared_content(a, b, placeholder="X"):
    ta, tb = tokenize(a), tokenize(b)
    shared = (set(ta) & set(tb)) - STOPWORDS
    def sub(text, toks):
        out = text
        for t in sorted(shared, key=len, reverse=True):
            out = re.sub(rf"\b{re.escape(t)}\b", placeholder, out, flags=re.I)
        return out
    return sub(a, ta), sub(b, tb)


def buckets_of(text):
    toks = set(tokenize(text))
    return {name for name, vocab in BUCKETS.items() if toks & vocab}


def entity_conflict(a, b):
    ba, bb = buckets_of(a), buckets_of(b)
    hit = []
    for pair in CONFLICT_PAIRS:
        x, y = tuple(pair)
        if (x in ba and y in bb) or (y in ba and x in bb):
            hit.append(tuple(sorted(pair)))
    names_a = {n.lower() for n in proper_nouns(a)}
    names_b = {n.lower() for n in proper_nouns(b)}
    shared_names = names_a & names_b
    return {
        "conflict": bool(hit),
        "pairs": hit,
        "shared_names": sorted(shared_names),
        "name_and_conflict": bool(hit) and bool(shared_names),
        "buckets_a": sorted(ba),
        "buckets_b": sorted(bb),
    }


def jaccard(a, b):
    sa, sb = set(tokenize(a)) - STOPWORDS, set(tokenize(b)) - STOPWORDS
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def summarize(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    s = np.sort(arr)
    n = len(s)
    def pct(p):
        return float(s[min(n - 1, int(round((n - 1) * p)))])
    mean = float(arr.mean())
    std = float(arr.std()) if n > 1 else 0.0
    return {
        "n": n, "mean": mean, "std": std, "min": float(s[0]), "max": float(s[-1]),
        "p05": pct(0.05), "p25": pct(0.25), "p50": pct(0.50), "p75": pct(0.75), "p95": pct(0.95),
    }


def auroc(pos, neg):
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return None
    wins = 0.0
    for p in pos:
        wins += float(np.sum(p > neg)) + 0.5 * float(np.sum(p == neg))
    return wins / (len(pos) * len(neg))


def cohens_d(pos, neg):
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if pos.size < 2 or neg.size < 2:
        return None
    pooled = math.sqrt(((pos.size - 1) * pos.var(ddof=1) + (neg.size - 1) * neg.var(ddof=1)) /
                       (pos.size + neg.size - 2))
    if pooled == 0:
        return None
    return float((pos.mean() - neg.mean()) / pooled)


def rf(x, nd=4):
    if x is None:
        return None
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return round(x, nd)
    return x


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = [{"id": d["id"], "terms": tokenize(d["text"])} for d in docs]
        self.avgdl = sum(len(d["terms"]) for d in self.docs) / max(1, len(self.docs))
        self.df = defaultdict(int)
        for d in self.docs:
            for t in set(d["terms"]):
                self.df[t] += 1
        self.N = len(self.docs)

    def idf(self, t):
        n = self.df.get(t, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def score_all(self, query):
        q = tokenize(query)
        out = []
        for d in self.docs:
            s = 0.0
            length = len(d["terms"])
            tf = defaultdict(int)
            for t in d["terms"]:
                tf[t] += 1
            for t in q:
                if tf[t] == 0:
                    continue
                denom = tf[t] + self.k1 * (1 - self.b + self.b * length / self.avgdl)
                s += self.idf(t) * (tf[t] * (self.k1 + 1)) / denom
            out.append((d["id"], s))
        out.sort(key=lambda x: -x[1])
        return out


# ---------------------------------------------------------------------------
# Model loading / encoding
# ---------------------------------------------------------------------------

def encode_st(model, texts, prompt=None, prompt_name=None, batch_size=16, prefix=None):
    if prefix:
        texts = [prefix + t for t in texts]
    kwargs = dict(show_progress_bar=False, normalize_embeddings=True, batch_size=batch_size)
    if prompt is not None:
        kwargs["prompt"] = prompt
    if prompt_name is not None:
        kwargs["prompt_name"] = prompt_name
    embs = model.encode(list(texts), **kwargs)
    arr = np.asarray(embs, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    # Belt-and-suspenders L2.
    n = np.linalg.norm(arr, axis=1, keepdims=True)
    n = np.clip(n, 1e-12, None)
    return arr / n


def cache_path(tag, split):
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / f"{tag}__{split}.npy"


def save_emb(tag, split, arr):
    np.save(cache_path(tag, split), arr)


def load_emb(tag, split):
    p = cache_path(tag, split)
    if p.exists():
        return np.load(p)
    return None


def get_or_encode(tag, split, texts, fn):
    cached = load_emb(tag, split)
    if cached is not None and cached.shape[0] == len(texts):
        log(f"    cache hit {tag}/{split} {cached.shape}")
        return cached
    log(f"    encoding {tag}/{split} n={len(texts)}")
    t0 = time.time()
    arr = fn(texts)
    save_emb(tag, split, arr)
    log(f"      -> {arr.shape} in {time.time()-t0:.1f}s")
    return arr


def load_models():
    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(max(4, os.cpu_count() or 8))
    models = {}

    log("Loading nomic-embed-text-v1.5 …")
    try:
        models["nomic"] = SentenceTransformer(
            str(NOMIC_PATH) if NOMIC_PATH.exists() else "nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True, device="cpu",
        )
        log(f"  nomic dim={models['nomic'].get_sentence_embedding_dimension()} prompts={getattr(models['nomic'], 'prompts', None)}")
    except Exception as e:
        log(f"  NOMIC LOAD FAILED: {e}")
        try:
            models["nomic"] = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device="cpu")
            log("  nomic loaded from hub fallback")
        except Exception as e2:
            log(f"  NOMIC HUB FALLBACK FAILED: {e2}")

    log("Loading Qwen3-Embedding-0.6B …")
    try:
        models["qwen"] = SentenceTransformer(str(QWEN_PATH), trust_remote_code=True, device="cpu")
        log(f"  qwen dim={models['qwen'].get_sentence_embedding_dimension()} prompts={list((models['qwen'].prompts or {}).keys())}")
    except Exception as e:
        log(f"  QWEN LOAD FAILED: {e}")

    log("Loading jina-embeddings-v5-text-nano-retrieval …")
    try:
        models["jina"] = SentenceTransformer(str(JINA_PATH), trust_remote_code=True, device="cpu")
        log(f"  jina dim={models['jina'].get_sentence_embedding_dimension()} prompts={list((models['jina'].prompts or {}).keys())}")
    except Exception as e:
        log(f"  JINA LOAD FAILED: {e}")
        import traceback
        traceback.print_exc()
    return models


def mode_specs(models):
    """Each spec: (tag, model_key, doc_fn, query_fn, notes, is_correct, geometry)."""
    specs = []

    if "nomic" in models:
        m = models["nomic"]
        specs.append(("nomic__plain", "nomic",
                      lambda ts, m=m: encode_st(m, ts, batch_size=32),
                      lambda ts, m=m: encode_st(m, ts, batch_size=32),
                      "PLAIN — reproduces the prior (wrong) comparison", False, "symmetric"))
        specs.append(("nomic__search_doc", "nomic",
                      lambda ts, m=m: encode_st(m, ts, prefix="search_document: ", batch_size=32),
                      lambda ts, m=m: encode_st(m, ts, prefix="search_document: ", batch_size=32),
                      "search_document: on both sides (symmetric doc geometry: field/dedup/save-bind)", True, "symmetric"))
        specs.append(("nomic__clustering", "nomic",
                      lambda ts, m=m: encode_st(m, ts, prefix="clustering: ", batch_size=32),
                      lambda ts, m=m: encode_st(m, ts, prefix="clustering: ", batch_size=32),
                      "clustering: on both sides (nomic STS/clustering task)", True, "symmetric"))
        specs.append(("nomic__asymmetric", "nomic",
                      lambda ts, m=m: encode_st(m, ts, prefix="search_document: ", batch_size=32),
                      lambda ts, m=m: encode_st(m, ts, prefix="search_query: ", batch_size=32),
                      "search_query: vs search_document: (asymmetric retrieval)", True, "asymmetric"))

    if "qwen" in models:
        m = models["qwen"]
        specs.append(("qwen__plain", "qwen",
                      lambda ts, m=m: encode_st(m, ts, batch_size=8),
                      lambda ts, m=m: encode_st(m, ts, batch_size=8),
                      "PLAIN — prior comparison", False, "symmetric"))
        specs.append(("qwen__default_query", "qwen",
                      lambda ts, m=m: encode_st(m, ts, batch_size=8),
                      lambda ts, m=m: encode_st(m, ts, prompt_name="query", batch_size=8),
                      "Official: encode(docs) + encode(queries, prompt_name='query')", True, "asymmetric"))
        specs.append(("qwen__sts_both", "qwen",
                      lambda ts, m=m: encode_st(m, ts, prompt=QWEN_STS, batch_size=8),
                      lambda ts, m=m: encode_st(m, ts, prompt=QWEN_STS, batch_size=8),
                      "STS instruction on both sides (symmetric relatedness)", True, "symmetric"))
        specs.append(("qwen__cluster_both", "qwen",
                      lambda ts, m=m: encode_st(m, ts, prompt=QWEN_CLUSTER, batch_size=8),
                      lambda ts, m=m: encode_st(m, ts, prompt=QWEN_CLUSTER, batch_size=8),
                      "Clustering instruction that asks to distinguish shared names", True, "symmetric"))
        specs.append(("qwen__memory_asym", "qwen",
                      lambda ts, m=m: encode_st(m, ts, batch_size=8),
                      lambda ts, m=m: encode_st(m, ts, prompt=QWEN_MEMORY, batch_size=8),
                      "Custom personal-memory retrieval instruction on queries only", True, "asymmetric"))

    if "jina" in models:
        m = models["jina"]
        specs.append(("jina__plain", "jina",
                      lambda ts, m=m: encode_st(m, ts, batch_size=8),
                      lambda ts, m=m: encode_st(m, ts, batch_size=8),
                      "PLAIN — prior comparison (the 0.05–0.87 smoking gun)", False, "symmetric"))
        specs.append(("jina__document_both", "jina",
                      lambda ts, m=m: encode_st(m, ts, prompt_name="document", batch_size=8),
                      lambda ts, m=m: encode_st(m, ts, prompt_name="document", batch_size=8),
                      "Document: prefix both sides (symmetric field/dedup)", True, "symmetric"))
        specs.append(("jina__asymmetric", "jina",
                      lambda ts, m=m: encode_st(m, ts, prompt_name="document", batch_size=8),
                      lambda ts, m=m: encode_st(m, ts, prompt_name="query", batch_size=8),
                      "Official: prompt_name query vs document", True, "asymmetric"))
    return specs


# ---------------------------------------------------------------------------
# Pairwise / retrieval / cofire
# ---------------------------------------------------------------------------

def pair_scores(id2emb, pairs):
    out = []
    for p in pairs:
        ea, eb = id2emb.get(p["a"]), id2emb.get(p["b"])
        if ea is None or eb is None:
            continue
        out.append({"a": p["a"], "b": p["b"], "theme_id": p.get("theme_id"), "cos": cosine(ea, eb)})
    return out


def theme_of_pair(corpus, a, b):
    recs = {r["id"]: r for r in corpus["records"]}
    ta = recs.get(a, {}).get("theme_id")
    tb = recs.get(b, {}).get("theme_id")
    return ta if ta == tb else None


def pairwise_block(corpus, rec_embs):
    id2emb = {r["id"]: rec_embs[i] for i, r in enumerate(corpus["records"])}
    true = pair_scores(id2emb, corpus["true_pairs"])
    nm = pair_scores(id2emb, corpus["near_miss_pairs"])
    un = pair_scores(id2emb, corpus["unrelated_pairs"])
    tcos = [x["cos"] for x in true]
    ncos = [x["cos"] for x in nm]
    ucos = [x["cos"] for x in un]
    by_theme = defaultdict(lambda: {"true": [], "nm": []})
    recs = {r["id"]: r for r in corpus["records"]}
    for x in true:
        th = recs[x["a"]]["theme_id"]
        by_theme[th]["true"].append(x)
    for x in nm:
        th = recs[x["a"]]["theme_id"]
        by_theme[th]["nm"].append(x)
    theme_rows = {}
    for th, d in by_theme.items():
        tm = [x["cos"] for x in d["true"]]
        nmv = [x["cos"] for x in d["nm"]]
        theme_rows[th] = {
            "true": [rf(c) for c in tm],
            "nm": [rf(c) for c in nmv],
            "true_mean": rf(float(np.mean(tm))) if tm else None,
            "nm_mean": rf(float(np.mean(nmv))) if nmv else None,
            "gap": rf(float(np.mean(tm) - np.mean(nmv))) if tm and nmv else None,
            "overlap": (min(tm) <= max(nmv)) if tm and nmv else None,
            "worst_true": rf(min(tm)) if tm else None,
            "best_nm": rf(max(nmv)) if nmv else None,
        }
    return {
        "true": summarize(tcos),
        "near_miss": summarize(ncos),
        "unrelated": summarize(ucos),
        "true_vs_nm_gap": rf(float(np.mean(tcos) - np.mean(ncos))) if tcos and ncos else None,
        "true_vs_un_gap": rf(float(np.mean(tcos) - np.mean(ucos))) if tcos and ucos else None,
        "nm_vs_un_gap": rf(float(np.mean(ncos) - np.mean(ucos))) if ncos and ucos else None,
        "true_nm_ratio": rf(float(np.mean(tcos) / np.mean(ncos))) if tcos and ncos and np.mean(ncos) else None,
        "auroc_true_nm": rf(auroc(tcos, ncos)),
        "auroc_true_un": rf(auroc(tcos, ucos)),
        "auroc_nm_un": rf(auroc(ncos, ucos)),
        "cohens_d_true_nm": rf(cohens_d(tcos, ncos)),
        "worst_true": rf(min(tcos)) if tcos else None,
        "best_nm": rf(max(ncos)) if ncos else None,
        "best_un": rf(max(ucos)) if ucos else None,
        "hard_sep_true_nm": bool(tcos and ncos and min(tcos) > max(ncos)),
        "hard_sep_true_un": bool(tcos and ucos and min(tcos) > max(ucos)),
        "per_pair_true": [{**x, "cos": rf(x["cos"])} for x in true],
        "per_pair_nm": [{**x, "cos": rf(x["cos"])} for x in nm],
        "per_theme": theme_rows,
        "_tcos": tcos, "_ncos": ncos, "_ucos": ucos,
    }


def retrieval_block(corpus, rec_embs, q_embs, k=5):
    rec_ids = [r["id"] for r in corpus["records"]]
    recs = corpus["records"]
    rec_by_id = {r["id"]: r for r in recs}
    nm_ids = {r["id"] for r in recs if r["role"] == "near-miss"}
    # Precompute sims: (nq, nr)
    S = q_embs @ rec_embs.T
    h1_hits = 0
    h1_para = 0
    h1_para_n = 0
    nm_in_topk = 0
    traces_nm = []
    cofire = defaultdict(int)
    nq = len(corpus["queries"])
    for qi, q in enumerate(corpus["queries"]):
        sims = S[qi]
        order = np.argsort(-sims)[:k]
        top_ids = [rec_ids[j] for j in order]
        target = q.get("target_id")
        hit = False
        if target:
            hit = target in top_ids
        elif q.get("theme_id"):
            members = [r["id"] for r in recs if r["theme_id"] == q["theme_id"] and r["role"] == "member"]
            hit = any(i in top_ids for i in members)
        if hit:
            h1_hits += 1
        if q.get("kind") == "member-paraphrase":
            h1_para_n += 1
            if hit:
                h1_para += 1
        leaked = [i for i in top_ids if i in nm_ids]
        if leaked:
            nm_in_topk += 1
            traces_nm.append({
                "q": q["query"][:80], "kind": q.get("kind"), "theme_id": q.get("theme_id"),
                "target": target, "leaked": leaked, "top": top_ids,
                "sims": [rf(float(sims[j])) for j in order],
            })
        for i in range(len(top_ids)):
            for j in range(i + 1, len(top_ids)):
                a, b = top_ids[i], top_ids[j]
                key = (a, b) if a < b else (b, a)
                cofire[key] += 1

    def rates(pairs):
        vals = []
        for p in pairs:
            a, b = p["a"], p["b"]
            key = (a, b) if a < b else (b, a)
            vals.append(cofire.get(key, 0) / nq)
        return vals

    t_rates = rates(corpus["true_pairs"])
    n_rates = rates(corpus["near_miss_pairs"])
    u_rates = rates(corpus["unrelated_pairs"])
    # Per-theme near-miss leak on that theme's own queries
    theme_leak = {}
    for theme in {r["theme_id"] for r in recs if r["theme_id"]}:
        q_idx = [i for i, q in enumerate(corpus["queries"]) if q.get("theme_id") == theme]
        nm_id = next((r["id"] for r in recs if r["theme_id"] == theme and r["role"] == "near-miss"), None)
        if not q_idx or not nm_id:
            continue
        leak = 0
        for i in q_idx:
            order = np.argsort(-S[i])[:k]
            top_ids = [rec_ids[j] for j in order]
            if nm_id in top_ids:
                leak += 1
        theme_leak[theme] = {"n": len(q_idx), "nm_in_topk": leak, "rate": rf(leak / len(q_idx))}

    return {
        "k": k,
        "h1_rate": rf(h1_hits / nq),
        "h1_paraphrase_rate": rf(h1_para / h1_para_n) if h1_para_n else None,
        "nm_in_topk_rate": rf(nm_in_topk / nq),
        "true_cofire_mean": rf(float(np.mean(t_rates))) if t_rates else None,
        "nm_cofire_mean": rf(float(np.mean(n_rates))) if n_rates else None,
        "un_cofire_mean": rf(float(np.mean(u_rates))) if u_rates else None,
        "true_nm_cofire_ratio": rf(float(np.mean(t_rates) / np.mean(n_rates))) if t_rates and n_rates and np.mean(n_rates) else None,
        "theme_nm_leak": theme_leak,
        "nm_leak_examples": traces_nm[:12],
        "_t_rates": t_rates, "_n_rates": n_rates, "_u_rates": u_rates,
        "_S": S,
    }


def threshold_scan(tcos, ncos, ucos, lo=0.05, hi=0.99, step=0.01):
    rows = []
    best = None
    t = np.asarray(tcos)
    n = np.asarray(ncos)
    u = np.asarray(ucos)
    x = lo
    while x <= hi + 1e-9:
        t_keep = float(np.mean(t >= x)) if t.size else 0
        n_leak = float(np.mean(n >= x)) if n.size else 0
        u_leak = float(np.mean(u >= x)) if u.size else 0
        # Youden-like: keep true, reject nm and un.
        score = t_keep - 0.5 * n_leak - 0.5 * u_leak
        row = {"t": rf(x), "true_keep": rf(t_keep), "nm_leak": rf(n_leak), "un_leak": rf(u_leak), "score": rf(score)}
        rows.append(row)
        if best is None or score > best["score"]:
            best = row
        x += step
    # Keep a coarse curve (every 0.05) plus the best point.
    coarse = [r for r in rows if abs(round(r["t"] / 0.05) * 0.05 - r["t"]) < 1e-9]
    return {"best": best, "curve": coarse}


def derive_presets(dup_groups, dup_controls, pairwise, recs, rec_embs, model_dim):
    """Presets from THIS model's geometry.

    dedup_hi / dedup_lo come from the duplicates corpus (restatement vs mid-band
    vs controls) — NOT from theme-member pairs. Theme members are related-but-
    distinct; merging them is a product bug.

    field_minsim comes from fire-together pairwise: sit above unrelated p95
    when possible, below true-related p05, and ideally above near-miss p95.
    """
    hi_cos, mid_cos, ctrl_cos = [], [], []
    for g, items in dup_groups.items():
        embs = None
        # caller passes a dict group -> list of (text, emb) via dup_groups_measured
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                c = cosine(items[i]["emb"], items[j]["emb"])
                band = items[i].get("band")
                if band in ("hi", "exact"):
                    hi_cos.append(c)
                elif band == "mid":
                    mid_cos.append(c)
    for i in range(len(dup_controls)):
        for j in range(i + 1, len(dup_controls)):
            ctrl_cos.append(cosine(dup_controls[i]["emb"], dup_controls[j]["emb"]))
    # Also controls vs everything in groups
    all_group_items = [it for items in dup_groups.values() for it in items]
    cross = []
    for c in dup_controls:
        for it in all_group_items:
            cross.append(cosine(c["emb"], it["emb"]))

    t = pairwise["_tcos"]
    n = pairwise["_ncos"]
    u = pairwise["_ucos"]
    t_s, n_s, u_s = summarize(t), summarize(n), summarize(u)
    hi_s, mid_s, ctrl_s = summarize(hi_cos), summarize(mid_cos), summarize(ctrl_cos + cross)

    # dedup_hi: below typical restatement, above mid-band. nomic production is 0.95
    # because tea restatement measured 0.9522. Use min(hi) *slightly below* so
    # restatements still catch, but never below mid max if they overlap.
    if hi_cos:
        dedup_hi = float(np.percentile(hi_cos, 5))  # almost all restatements at or above
        # If mid-band exists, keep hi above mid p50 so mid stays merge-not-restatement.
        if mid_cos:
            dedup_hi = max(dedup_hi, float(np.percentile(mid_cos, 75)))
        dedup_hi = min(0.99, max(0.70, dedup_hi))
    else:
        dedup_hi = 0.95
    if mid_cos:
        dedup_lo = float(np.percentile(mid_cos, 5))
        # Must stay above control/cross so unrelated don't merge.
        if ctrl_s.get("p95"):
            dedup_lo = max(dedup_lo, ctrl_s["p95"] + 0.02)
        dedup_lo = min(dedup_hi - 0.02, max(0.50, dedup_lo))
    else:
        dedup_lo = 0.88

    # field_minsim: above unrelated p95, below true p05 if possible.
    un_p95 = u_s["p95"]
    true_p05 = t_s["p05"]
    nm_p95 = n_s["p95"]
    # Prefer a value that keeps ALL unrelated out and MOST true in.
    field = un_p95 + 0.02
    if field > true_p05:
        # overlap with true — sit in the middle of the gap between un p95 and true p50
        field = 0.5 * (un_p95 + t_s["p50"])
    field = min(0.90, max(0.20, field))

    # constraint_gate is a WEAKER rescue floor (currently 0.45 vs field 0.55).
    constraint = max(0.15, field - 0.10)

    # save_bind is a persist floor, deliberately looser than field (currently 0.25).
    # Sit near unrelated p05 so we persist weak-but-real neighbors, not noise.
    save_bind = max(0.10, min(field - 0.20, u_s["p05"] + 0.02))

    # Safety flags
    safety = {
        "dedup_hi_catches_hi_band": rf(float(np.mean(np.asarray(hi_cos) >= dedup_hi))) if hi_cos else None,
        "dedup_lo_below_controls_p95": bool(ctrl_s.get("p95") is not None and dedup_lo > ctrl_s["p95"]),
        "field_un_leak": rf(float(np.mean(np.asarray(u) >= field))),
        "field_true_keep": rf(float(np.mean(np.asarray(t) >= field))),
        "field_nm_leak": rf(float(np.mean(np.asarray(n) >= field))),
        "note": "field_nm_leak > 0 means near-misses still enter Related: at this gate — that is the source problem, not a calibration bug.",
    }
    return {
        "dedup_hi": rf(dedup_hi, 4),
        "dedup_lo": rf(dedup_lo, 4),
        "field_minsim": rf(field, 4),
        "constraint_gate": rf(constraint, 4),
        "save_bind": rf(save_bind, 4),
        "dim": int(model_dim),
        "dup_hi": summarize(hi_cos),
        "dup_mid": summarize(mid_cos),
        "dup_ctrl_cross": summarize(ctrl_cos + cross),
        "safety": safety,
        "scan": threshold_scan(t, n, u),
    }


def hybrid_pairwise(corpus, rec_embs):
    recs = corpus["records"]
    id2 = {r["id"]: (r, rec_embs[i]) for i, r in enumerate(recs)}
    rows = []
    for kind, pairs in (("true", corpus["true_pairs"]), ("near_miss", corpus["near_miss_pairs"]),
                        ("unrelated", corpus["unrelated_pairs"])):
        for p in pairs:
            ra, ea = id2[p["a"]]
            rb, eb = id2[p["b"]]
            cos = cosine(ea, eb)
            jac = jaccard(ra["text"], rb["text"])
            conf = entity_conflict(ra["text"], rb["text"])
            penalty = 0.0
            if conf["name_and_conflict"]:
                penalty = 0.25
            elif conf["conflict"]:
                penalty = 0.15
            rows.append({
                "kind": kind, "a": p["a"], "b": p["b"],
                "cos": cos, "jaccard": jac,
                "conflict": conf["conflict"],
                "name_and_conflict": conf["name_and_conflict"],
                "shared_names": conf["shared_names"],
                "hybrid_sub": cos - penalty,
                "hybrid_jac": 0.7 * cos + 0.3 * jac,
                "hybrid_and": (0.0 if conf["name_and_conflict"] else cos),
            })
    by = defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r)

    def pack(key):
        out = {}
        for kind in ("true", "near_miss", "unrelated"):
            vals = [r[key] for r in by[kind]]
            out[kind] = summarize(vals)
        t = [r[key] for r in by["true"]]
        n = [r[key] for r in by["near_miss"]]
        u = [r[key] for r in by["unrelated"]]
        out["auroc_true_nm"] = rf(auroc(t, n))
        out["auroc_true_un"] = rf(auroc(t, u))
        out["true_nm_ratio"] = rf(float(np.mean(t) / np.mean(n))) if n and np.mean(n) else None
        out["true_vs_nm_gap"] = rf(float(np.mean(t) - np.mean(n)))
        out["un_max"] = rf(max(u) if u else None)
        out["nm_max"] = rf(max(n) if n else None)
        out["true_min"] = rf(min(t) if t else None)
        # Honest safety: unrelated must not inflate
        out["un_mean"] = rf(float(np.mean(u))) if u else None
        return out

    # Per-confound hybrid on labeled near-misses
    nm_detail = []
    for r in by["near_miss"]:
        nm_detail.append({
            "a": r["a"], "b": r["b"], "cos": rf(r["cos"]),
            "hybrid_sub": rf(r["hybrid_sub"]),
            "hybrid_and": rf(r["hybrid_and"]),
            "jaccard": rf(r["jaccard"]),
            "conflict": r["conflict"],
            "name_and_conflict": r["name_and_conflict"],
            "shared_names": r["shared_names"],
        })
    return {
        "cosine": pack("cos"),
        "hybrid_sub": pack("hybrid_sub"),
        "hybrid_jac": pack("hybrid_jac"),
        "hybrid_and": pack("hybrid_and"),
        "nm_detail": nm_detail,
        "true_conflict_false_pos": sum(1 for r in by["true"] if r["conflict"]),
        "un_conflict_count": sum(1 for r in by["unrelated"] if r["conflict"]),
    }


def rrf_retrieval(corpus, rec_embs, q_embs, k=5, rrf_k=60):
    recs = corpus["records"]
    rec_ids = [r["id"] for r in recs]
    bm25 = BM25(recs)
    S = q_embs @ rec_embs.T
    nq = len(corpus["queries"])
    h1 = 0
    nm_ids = {r["id"] for r in recs if r["role"] == "near-miss"}
    nm_in = 0
    cofire = defaultdict(int)
    for qi, q in enumerate(corpus["queries"]):
        dense_order = list(np.argsort(-S[qi]))
        sparse = bm25.score_all(q["query"])
        sparse_rank = {did: i for i, (did, _) in enumerate(sparse)}
        scores = {}
        for rank, idx in enumerate(dense_order):
            scores[rec_ids[idx]] = scores.get(rec_ids[idx], 0) + 1.0 / (rrf_k + rank + 1)
        for did, rnk in sparse_rank.items():
            scores[did] = scores.get(did, 0) + 1.0 / (rrf_k + rnk + 1)
        top = sorted(scores.items(), key=lambda x: -x[1])[:k]
        top_ids = [t[0] for t in top]
        target = q.get("target_id")
        hit = False
        if target:
            hit = target in top_ids
        elif q.get("theme_id"):
            members = [r["id"] for r in recs if r["theme_id"] == q["theme_id"] and r["role"] == "member"]
            hit = any(i in top_ids for i in members)
        if hit:
            h1 += 1
        if any(i in nm_ids for i in top_ids):
            nm_in += 1
        for i in range(len(top_ids)):
            for j in range(i + 1, len(top_ids)):
                a, b = top_ids[i], top_ids[j]
                key = (a, b) if a < b else (b, a)
                cofire[key] += 1

    def mean_rate(pairs):
        vals = []
        for p in pairs:
            a, b = p["a"], p["b"]
            key = (a, b) if a < b else (b, a)
            vals.append(cofire.get(key, 0) / nq)
        return float(np.mean(vals)) if vals else 0.0

    t, n, u = mean_rate(corpus["true_pairs"]), mean_rate(corpus["near_miss_pairs"]), mean_rate(corpus["unrelated_pairs"])
    return {
        "h1_rate": rf(h1 / nq),
        "nm_in_topk_rate": rf(nm_in / nq),
        "true_cofire_mean": rf(t),
        "nm_cofire_mean": rf(n),
        "un_cofire_mean": rf(u),
        "true_nm_cofire_ratio": rf(t / n if n else None),
    }


def reinforce_sim(corpus, rec_embs, q_embs, k=5, sessions=10, weighted=False, field_minsim=0.55):
    """Replay theme-cycled sessions. Weight update matches EdgeStore: w += alphaPP (0.1).
    bonus = maxBonus * tanh(w). Optional: scale alpha by cosine so a 0.65 near-miss
    earns a smaller bump than a 0.84 true pair."""
    recs = corpus["records"]
    rec_ids = [r["id"] for r in recs]
    rec_by = {r["id"]: i for i, r in enumerate(recs)}
    S = q_embs @ rec_embs.T
    themes = [t["id"] for t in corpus["themes"]]
    weights = defaultdict(float)
    ALPHA = 0.1
    MAX_BONUS = 0.3

    def key(a, b):
        return (a, b) if a < b else (b, a)

    def bump(ids, sims_by_id):
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                alpha = ALPHA
                if weighted:
                    # geometric mean of the two docs' query sims, floored at 0
                    sa = max(0.0, sims_by_id[a])
                    sb = max(0.0, sims_by_id[b])
                    # also scale by doc-doc cosine so near-miss (lower cos) earns less
                    ea = rec_embs[rec_by[a]]
                    eb = rec_embs[rec_by[b]]
                    dd = max(0.0, cosine(ea, eb))
                    # map [field_minsim, 1] -> [0, 1]; below gate earns nothing
                    gate = max(0.0, (dd - field_minsim) / max(1e-6, 1.0 - field_minsim))
                    alpha = ALPHA * gate * math.sqrt(sa * sb)
                weights[key(a, b)] += alpha

    q_by_theme = defaultdict(list)
    for i, q in enumerate(corpus["queries"]):
        if q.get("kind") == "member-paraphrase" and q.get("theme_id"):
            q_by_theme[q["theme_id"]].append(i)

    history = []
    for sess in range(sessions):
        for th in themes:
            for qi in q_by_theme.get(th, []):
                sims = S[qi]
                order = np.argsort(-sims)[:k]
                top_ids = [rec_ids[j] for j in order]
                sims_by_id = {rec_ids[j]: float(sims[j]) for j in order}
                bump(top_ids, sims_by_id)

        def mean_w(pairs):
            vals = [weights[key(p["a"], p["b"])] for p in pairs]
            return float(np.mean(vals)) if vals else 0.0

        def mean_bonus(pairs):
            vals = [MAX_BONUS * math.tanh(weights[key(p["a"], p["b"])]) for p in pairs]
            return float(np.mean(vals)) if vals else 0.0

        def mean_norm_bonus(pairs):
            # divide-by-node-max readout
            node_max = defaultdict(float)
            for (a, b), w in weights.items():
                node_max[a] = max(node_max[a], w)
                node_max[b] = max(node_max[b], w)
            vals = []
            for p in pairs:
                w = weights[key(p["a"], p["b"])]
                mx = max(node_max[p["a"]], node_max[p["b"]], 1e-9)
                vals.append(MAX_BONUS * math.tanh(w / mx))
            return float(np.mean(vals)) if vals else 0.0

        tw, nw, uw = mean_w(corpus["true_pairs"]), mean_w(corpus["near_miss_pairs"]), mean_w(corpus["unrelated_pairs"])
        tb, nb, ub = mean_bonus(corpus["true_pairs"]), mean_bonus(corpus["near_miss_pairs"]), mean_bonus(corpus["unrelated_pairs"])
        tn, nn, un = mean_norm_bonus(corpus["true_pairs"]), mean_norm_bonus(corpus["near_miss_pairs"]), mean_norm_bonus(corpus["unrelated_pairs"])
        history.append({
            "session": sess,
            "w_true": rf(tw), "w_nm": rf(nw), "w_un": rf(uw),
            "w_ratio": rf(tw / nw if nw else None),
            "bonus_true": rf(tb), "bonus_nm": rf(nb), "bonus_un": rf(ub),
            "bonus_ratio": rf(tb / nb if nb else None),
            "norm_true": rf(tn), "norm_nm": rf(nn), "norm_un": rf(un),
            "norm_ratio": rf(tn / nn if nn else None),
        })
    return history


def adapter_loto(corpus, rec_embs, q_embs, dim, rank=8, steps=250, seed=0):
    """Leave-one-theme-out low-rank residual projection. Tiny on purpose.
    If held-out AUROC does not rise, the geometry is not linearly fixable
    from this label set — that is a result, not a failure of training."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    recs = corpus["records"]
    themes = [t["id"] for t in corpus["themes"]]
    rec_ids = [r["id"] for r in recs]
    id2i = {r["id"]: i for i, r in enumerate(recs)}
    members = defaultdict(list)
    nm_of = {}
    for r in recs:
        if r["role"] == "member" and r["theme_id"]:
            members[r["theme_id"]].append(r["id"])
        if r["role"] == "near-miss" and r["theme_id"]:
            nm_of[r["theme_id"]] = r["id"]

    X = torch.tensor(rec_embs, dtype=torch.float32)
    Q = torch.tensor(q_embs, dtype=torch.float32)

    folds = []
    for held in themes:
        train_themes = [t for t in themes if t != held]
        # triplets: (anchor_idx, pos_idx, neg_idx) over records + query anchors
        trips = []
        for th in train_themes:
            mids = members[th]
            neg = nm_of.get(th)
            if not neg or len(mids) < 2:
                continue
            for i, a in enumerate(mids):
                for p in mids:
                    if p == a:
                        continue
                    trips.append((id2i[a], id2i[p], id2i[neg]))
            # query anchors (member-paraphrase of this theme)
            for qi, q in enumerate(corpus["queries"]):
                if q.get("kind") != "member-paraphrase" or q.get("theme_id") != th:
                    continue
                tgt = q.get("target_id")
                if not tgt or tgt not in id2i:
                    continue
                # positive = target, negative = theme near-miss
                trips.append(("q", qi, id2i[tgt], id2i[neg]))
        if not trips:
            continue

        torch.manual_seed(seed)
        A = nn.Parameter(torch.zeros(dim, rank))
        B = nn.Parameter(torch.randn(rank, dim) * 0.01)
        opt = torch.optim.Adam([A, B], lr=5e-3)
        rng = random.Random(seed + 17)

        def proj(v):
            return F.normalize(v + v @ A @ B, dim=-1)

        for step in range(steps):
            batch = [trips[rng.randrange(len(trips))] for _ in range(32)]
            loss = 0.0
            n = 0
            for item in batch:
                if item[0] == "q":
                    _, qi, pi, ni = item
                    a = proj(Q[qi])
                    p = proj(X[pi])
                    negv = proj(X[ni])
                else:
                    ai, pi, ni = item
                    a = proj(X[ai])
                    p = proj(X[pi])
                    negv = proj(X[ni])
                loss = loss + F.relu(0.2 - (a * p).sum() + (a * negv).sum())
                n += 1
            loss = loss / max(1, n)
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            Xp = proj(X).cpu().numpy()
        # Held-out pairwise AUROC
        tcos, ncos = [], []
        for p in corpus["true_pairs"]:
            if recs[id2i[p["a"]]]["theme_id"] == held:
                tcos.append(cosine(Xp[id2i[p["a"]]], Xp[id2i[p["b"]]]))
        for p in corpus["near_miss_pairs"]:
            if recs[id2i[p["a"]]]["theme_id"] == held:
                ncos.append(cosine(Xp[id2i[p["a"]]], Xp[id2i[p["b"]]]))
        ucos = []
        for p in corpus["unrelated_pairs"]:
            ucos.append(cosine(Xp[id2i[p["a"]]], Xp[id2i[p["b"]]]))
        folds.append({
            "held": held,
            "true_mean": rf(float(np.mean(tcos)) if tcos else None),
            "nm_mean": rf(float(np.mean(ncos)) if ncos else None),
            "un_mean": rf(float(np.mean(ucos)) if ucos else None),
            "auroc_true_nm": rf(auroc(tcos, ncos)),
            "un_max": rf(max(ucos) if ucos else None),
            "true_nm_gap": rf(float(np.mean(tcos) - np.mean(ncos)) if tcos and ncos else None),
        })
    return folds


def probe_block(probe_texts, probe_embs):
    # probe_texts aligned with PROBES flattened [a,b,a,b,...]
    out = []
    for i, p in enumerate(PROBES):
        ea, eb = probe_embs[2 * i], probe_embs[2 * i + 1]
        c = cosine(ea, eb)
        conf = entity_conflict(p["a"], p["b"])
        out.append({
            "id": p["id"], "cls": p["cls"], "expect": p["expect"],
            "cos": rf(c),
            "conflict": conf["conflict"],
            "name_and_conflict": conf["name_and_conflict"],
            "shared_names": conf["shared_names"],
            "a": p["a"], "b": p["b"],
        })
    by_cls = defaultdict(list)
    for r in out:
        by_cls[r["cls"]].append(r)
    summary = {cls: {"mean": rf(float(np.mean([r["cos"] for r in rs]))),
                     "values": [{k: r[k] for k in ("id", "expect", "cos", "name_and_conflict")} for r in rs]}
               for cls, rs in by_cls.items()}
    return {"pairs": out, "by_class": summary}


def name_mask_block(corpus, encode_docs):
    """Re-embed true + near-miss pairs with (1) proper-noun mask (2) shared-content mask."""
    recs = {r["id"]: r for r in corpus["records"]}
    pairs = [("true", p) for p in corpus["true_pairs"]] + [("nm", p) for p in corpus["near_miss_pairs"]]
    texts_pn = []
    texts_sh = []
    meta = []
    for kind, p in pairs:
        a, b = recs[p["a"]]["text"], recs[p["b"]]["text"]
        texts_pn += [mask_proper_nouns(a), mask_proper_nouns(b)]
        sa, sb = mask_shared_content(a, b)
        texts_sh += [sa, sb]
        meta.append((kind, p, a, b, sa, sb))
    em_pn = encode_docs(texts_pn)
    em_sh = encode_docs(texts_sh)
    rows = []
    for i, (kind, p, a, b, sa, sb) in enumerate(meta):
        rows.append({
            "kind": kind, "a": p["a"], "b": p["b"],
            "theme": recs[p["a"]]["theme_id"],
            "pn_mask_cos": rf(cosine(em_pn[2 * i], em_pn[2 * i + 1])),
            "shared_mask_cos": rf(cosine(em_sh[2 * i], em_sh[2 * i + 1])),
            "masked_shared_a": sa, "masked_shared_b": sb,
        })
    def pack(kind, key):
        vals = [r[key] for r in rows if r["kind"] == kind]
        return summarize(vals)
    return {
        "pn_true": pack("true", "pn_mask_cos"),
        "pn_nm": pack("nm", "pn_mask_cos"),
        "sh_true": pack("true", "shared_mask_cos"),
        "sh_nm": pack("nm", "shared_mask_cos"),
        "rows": rows,
    }


def json_ready(obj):
    if isinstance(obj, dict):
        return {k: json_ready(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [json_ready(x) for x in obj]
    if isinstance(obj, (np.floating,)):
        return rf(float(obj))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float):
        return rf(obj)
    return obj


def write_report(results):
    lines = []
    a = lines.append
    a("# Fair embedder comparison — invoked correctly")
    a("")
    a("**Date:** 2026-09-06")
    a("**Corpus:** fire-together 25 records (15 members + 5 near-miss + 5 distractors), 525 queries; duplicates.jsonl for dedup bands; 18 extra probes; 300-item haystack for cofire-with-noise.")
    a("**Rule:** each model is used the way its authors specify. The prior `fire-together-embedder-comparison.py` encoded everything PLAIN and is an artifact.")
    a("")
    a("## Headline")
    a("")
    # Build comparison table from correct-mode pairwise + retrieval
    a("| tag | geometry | true mean | nm mean | un mean | true/nm AUROC | true/nm ratio | H1@5 | nm-in-top5 | cofire true/nm |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for tag, r in results["modes"].items():
        pw = r["pairwise"]
        rv = r["retrieval"]
        a(f"| `{tag}` | {r['geometry']} | {pw['true']['mean']:.3f} | {pw['near_miss']['mean']:.3f} | {pw['unrelated']['mean']:.3f} | {pw['auroc_true_nm']} | {pw['true_nm_ratio']} | {rv['h1_rate']} | {rv['nm_in_topk_rate']} | {rv['true_nm_cofire_ratio']} |")
    a("")
    a("AUROC 0.5 = coin flip, 1.0 = every true pair outranks every near-miss. Ratio is mean_true/mean_nm (the 1.47x analog). Cofire ratio is the associative-layer source signal.")
    a("")
    a("## Per-theme (correct invocation only)")
    a("")
    for tag, r in results["modes"].items():
        if not r.get("is_correct"):
            continue
        a(f"### `{tag}`")
        a("")
        a("| theme | true mean | nm mean | gap | overlap? | worst true | best nm |")
        a("|---|---|---|---|---|---|---|")
        for th, row in r["pairwise"]["per_theme"].items():
            a(f"| {th} | {row['true_mean']} | {row['nm_mean']} | {row['gap']} | {row['overlap']} | {row['worst_true']} | {row['best_nm']} |")
        a("")
        a("Near-miss leak on that theme's own queries:")
        a("")
        for th, row in r["retrieval"]["theme_nm_leak"].items():
            a(f"- **{th}**: {row['nm_in_topk']}/{row['n']} queries put the near-miss in top-5 (rate {row['rate']})")
        a("")
    a("## Calibrated presets")
    a("")
    a("Derived from each model's own duplicates.jsonl (dedup bands) and fire-together pairwise (field/save-bind). Theme-member pairs are **not** restatements — the prior run's `dedup_hi = true p95` was the wrong statistic.")
    a("")
    a("```json")
    presets = {tag: r["presets"] for tag, r in results["modes"].items() if r.get("is_correct") and "presets" in r}
    # strip bulky scan curves from the fenced block
    slim = {}
    for tag, p in presets.items():
        slim[tag] = {k: p[k] for k in ("dedup_hi", "dedup_lo", "field_minsim", "constraint_gate", "save_bind", "dim")}
        slim[tag]["_safety"] = p.get("safety")
    a(json.dumps(slim, indent=2))
    a("```")
    a("")
    a("Panel shape (one object per embedder): `{ dedup_hi, dedup_lo, field_minsim, constraint_gate, save_bind, dim }`.")
    a("")
    if "hybrid" in results:
        a("## Hybrid lexical + entity")
        a("")
        a("Cosine vs cosine-minus-conflict vs cosine-AND-not-(shared-name-and-conflict) vs 0.7 cosine + 0.3 Jaccard.")
        a("")
        for tag, h in results["hybrid"].items():
            a(f"### `{tag}`")
            a("")
            a("| scheme | true mean | nm mean | un mean | AUROC true/nm | ratio | un max |")
            a("|---|---|---|---|---|---|---|")
            for name in ("cosine", "hybrid_sub", "hybrid_and", "hybrid_jac"):
                s = h[name]
                a(f"| {name} | {s['true']['mean']:.3f} | {s['near_miss']['mean']:.3f} | {s['unrelated']['mean']:.3f} | {s['auroc_true_nm']} | {s['true_nm_ratio']} | {s['un_max']} |")
            a("")
            a(f"False conflict on true pairs: {h['true_conflict_false_pos']}. Conflicts on unrelated: {h['un_conflict_count']}.")
            a("")
            a("Near-miss detail:")
            a("")
            for d in h["nm_detail"]:
                a(f"- {d['a']} ↔ {d['b']}: cos={d['cos']} hybrid_sub={d['hybrid_sub']} hybrid_and={d['hybrid_and']} conflict={d['conflict']} name+conflict={d['name_and_conflict']} names={d['shared_names']}")
            a("")
        if "rrf" in results:
            a("### RRF (dense + BM25) retrieval")
            a("")
            a("| tag | H1@5 | nm-in-top5 | cofire true/nm | un cofire |")
            a("|---|---|---|---|---|")
            for tag, r in results["rrf"].items():
                a(f"| `{tag}` | {r['h1_rate']} | {r['nm_in_topk_rate']} | {r['true_nm_cofire_ratio']} | {r['un_cofire_mean']} |")
            a("")
    if "name_mask" in results:
        a("## Name / shared-token ablation")
        a("")
        a("If masking the shared proper noun (or all shared content tokens) drops near-miss cosine while true pairs stay high, the confound is lexical/entity, not deep semantics. If both drop, the structure itself is the geometry.")
        a("")
        for tag, m in results["name_mask"].items():
            a(f"### `{tag}`")
            a("")
            a(f"- proper-noun mask: true mean={m['pn_true'].get('mean')} nm mean={m['pn_nm'].get('mean')}")
            a(f"- shared-content mask: true mean={m['sh_true'].get('mean')} nm mean={m['sh_nm'].get('mean')}")
            a("")
            a("| pair | kind | theme | pn-mask cos | shared-mask cos |")
            a("|---|---|---|---|---|")
            for row in m["rows"]:
                a(f"| {row['a']}↔{row['b']} | {row['kind']} | {row['theme']} | {row['pn_mask_cos']} | {row['shared_mask_cos']} |")
            a("")
    if "probes" in results:
        a("## Extra confound probes")
        a("")
        a("Held-out sentences (not the fire-together members, except a few restated for calibration). Classes: shared-name-conflict-relation (Naima-class), shared-name-same-relation, same-relation-different-name, shared-category, restatement, unrelated, structure-only.")
        a("")
        for tag, p in results["probes"].items():
            a(f"### `{tag}`")
            a("")
            a("| id | class | expect | cosine | name+conflict |")
            a("|---|---|---|---|---|")
            for row in p["pairs"]:
                a(f"| {row['id']} | {row['cls']} | {row['expect']} | {row['cos']} | {row['name_and_conflict']} |")
            a("")
    if "adapter" in results:
        a("## Leave-one-theme-out linear adapter")
        a("")
        a("Low-rank residual (rank 8) trained with triplet loss on 4 themes, tested on the held-out theme. If held-out AUROC does not beat the frozen embedder, a linear map cannot separate that confound from this label set.")
        a("")
        for tag, folds in results["adapter"].items():
            a(f"### `{tag}`")
            a("")
            a("| held-out theme | true mean | nm mean | un mean | AUROC | un max |")
            a("|---|---|---|---|---|---|")
            for f in folds:
                a(f"| {f['held']} | {f['true_mean']} | {f['nm_mean']} | {f['un_mean']} | {f['auroc_true_nm']} | {f['un_max']} |")
            a("")
    if "reinforce" in results:
        a("## Similarity-weighted reinforcement")
        a("")
        a("10 theme-cycled sessions of member-paraphrase queries, top-5 co-activate. Equal bump (current EdgeStore α=0.1) vs cosine-gated bump. Readout: raw weight ratio, tanh bonus ratio, divide-by-node-max bonus ratio. Unrelated must stay low.")
        a("")
        for tag, block in results["reinforce"].items():
            a(f"### `{tag}`")
            a("")
            for kind in ("equal", "weighted"):
                hist = block[kind]
                last = hist[-1]
                a(f"**{kind}** final: weight ratio {last['w_ratio']} · tanh bonus ratio {last['bonus_ratio']} · node-max-norm ratio {last['norm_ratio']} · un weight {last['w_un']}")
            a("")
    a("## Notes — tried, rejected, self-challenge")
    a("")
    notes = results.get("notes") or []
    for n in notes:
        a(f"- {n}")
    a("")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"Wrote {OUT_MD}")


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus()
    dup_groups, dup_controls = load_duplicates()
    haystack = generate_haystack(300, seed=42)
    probe_texts = []
    for p in PROBES:
        probe_texts += [p["a"], p["b"]]
    rec_texts = [r["text"] for r in corpus["records"]]
    q_texts = [q["query"] for q in corpus["queries"]]
    dup_items_flat = []
    for g, items in dup_groups.items():
        for it in items:
            dup_items_flat.append(it)
    dup_ctrl_texts = [c["text"] for c in dup_controls]
    dup_grp_texts = [it["text"] for it in dup_items_flat]
    hay_texts = [h["text"] for h in haystack]

    log(f"corpus records={len(rec_texts)} queries={len(q_texts)} dups={len(dup_grp_texts)} ctrl={len(dup_ctrl_texts)} probes={len(probe_texts)} hay={len(hay_texts)}")

    models = load_models()
    if not models:
        log("No models loaded.")
        sys.exit(1)
    specs = mode_specs(models)
    log(f"{len(specs)} encode modes")

    results = {"modes": {}, "hybrid": {}, "rrf": {}, "name_mask": {}, "probes": {}, "adapter": {}, "reinforce": {}, "notes": []}
    encoded = {}  # tag -> {records, queries, ...}

    for tag, mkey, doc_fn, query_fn, notes, is_correct, geometry in specs:
        log(f"\n=== {tag} ===\n  {notes}")
        rec_e = get_or_encode(tag, "records", rec_texts, doc_fn)
        q_e = get_or_encode(tag, "queries", q_texts, query_fn)
        dup_e = get_or_encode(tag, "dups", dup_grp_texts, doc_fn)
        ctrl_e = get_or_encode(tag, "dupctrl", dup_ctrl_texts, doc_fn)
        probe_e = get_or_encode(tag, "probes", probe_texts, doc_fn)
        encoded[tag] = {"records": rec_e, "queries": q_e, "dups": dup_e, "ctrl": ctrl_e, "probes": probe_e,
                        "doc_fn": doc_fn, "query_fn": query_fn}

        pw = pairwise_block(corpus, rec_e)
        rv = retrieval_block(corpus, rec_e, q_e, k=5)

        # attach dups for presets
        groups_m = defaultdict(list)
        for it, emb in zip(dup_items_flat, dup_e):
            groups_m[it["group"]].append({**it, "emb": emb})
        controls_m = [{**c, "emb": e} for c, e in zip(dup_controls, ctrl_e)]
        presets = derive_presets(groups_m, controls_m, pw, corpus["records"], rec_e, rec_e.shape[1])

        results["modes"][tag] = {
            "model": mkey,
            "notes": notes,
            "is_correct": is_correct,
            "geometry": geometry,
            "dim": int(rec_e.shape[1]),
            "pairwise": json_ready(pw),
            "retrieval": json_ready(rv),
            "presets": json_ready(presets),
        }
        log(f"  true={pw['true']['mean']:.3f} nm={pw['near_miss']['mean']:.3f} un={pw['unrelated']['mean']:.3f} "
            f"AUROC={pw['auroc_true_nm']} H1={rv['h1_rate']} nm@5={rv['nm_in_topk_rate']} cofire_ratio={rv['true_nm_cofire_ratio']}")

        results["probes"][tag] = probe_block(probe_texts, probe_e)

    # Deeper analysis on CORRECT modes only (and plain, as the artifact baseline).
    deep_tags = [tag for tag, r in results["modes"].items() if r["is_correct"] or tag.endswith("__plain")]
    # Prefer one correct symmetric + one correct asymmetric per model, plus plains.
    for tag in deep_tags:
        rec_e = encoded[tag]["records"]
        q_e = encoded[tag]["queries"]
        log(f"\n--- hybrid/RRF/mask {tag} ---")
        results["hybrid"][tag] = json_ready(hybrid_pairwise(corpus, rec_e))
        results["rrf"][tag] = json_ready(rrf_retrieval(corpus, rec_e, q_e, k=5))
        results["name_mask"][tag] = json_ready(name_mask_block(corpus, encoded[tag]["doc_fn"]))

    # Adapter + reinforce: the single best-looking CORRECT mode per model, plus nomic plain (production).
    candidates = []
    for tag, r in results["modes"].items():
        if r["is_correct"]:
            candidates.append((tag, r["pairwise"].get("auroc_true_nm") or 0, r["retrieval"].get("true_nm_cofire_ratio") or 0))
    # pick max AUROC then cofire ratio per model
    by_model = defaultdict(list)
    for tag, au, cr in candidates:
        by_model[results["modes"][tag]["model"]].append((tag, au, cr))
    chosen = []
    for m, lst in by_model.items():
        lst.sort(key=lambda x: (x[1], x[2]), reverse=True)
        chosen.append(lst[0][0])
    if "nomic__plain" in results["modes"]:
        chosen.append("nomic__plain")
    chosen = list(dict.fromkeys(chosen))
    log(f"\nDeep-train tags: {chosen}")

    for tag in chosen:
        rec_e = encoded[tag]["records"]
        q_e = encoded[tag]["queries"]
        dim = rec_e.shape[1]
        field = results["modes"][tag]["presets"]["field_minsim"]
        log(f"adapter LOTO {tag}")
        try:
            results["adapter"][tag] = adapter_loto(corpus, rec_e, q_e, dim)
        except Exception as e:
            log(f"  adapter failed: {e}")
            results["adapter"][tag] = {"error": str(e)}
        log(f"reinforce {tag}")
        results["reinforce"][tag] = {
            "equal": reinforce_sim(corpus, rec_e, q_e, weighted=False, field_minsim=field),
            "weighted": reinforce_sim(corpus, rec_e, q_e, weighted=True, field_minsim=field),
        }

    # Haystack cofire for chosen tags (correct invocation under noise)
    results["haystack"] = {}
    for tag in chosen:
        log(f"haystack encode {tag}")
        hay_e = get_or_encode(tag, "haystack", hay_texts, encoded[tag]["doc_fn"])
        rec_e = encoded[tag]["records"]
        q_e = encoded[tag]["queries"]
        all_recs = corpus["records"] + haystack
        all_emb = np.vstack([rec_e, hay_e])
        # retrieval on expanded store
        tmp_corpus = {
            **corpus,
            "records": all_recs,
            # pairs stay on the original ids
        }
        results["haystack"][tag] = json_ready(retrieval_block(tmp_corpus, all_emb, q_e, k=5))

    results["notes"] = [
        "Prior comparison encoded all models with model.encode(texts, normalize=True) and no prefixes/roles. Jina true-pair cosine ranging 0.05–0.87 was the smoking gun of wrong usage.",
        "Production RM (server.js) currently POSTs raw text to /v1/embeddings with no prefixes. nomic__plain is therefore the production geometry; prefixed nomic is a product change, not a free upgrade.",
        "dedup_hi/lo are calibrated on eval/corpora/duplicates.jsonl (restatement vs mid-band vs controls), not on theme-member pairs. Theme members are related-but-distinct; using their p95 as dedup_hi would merge distinct facts.",
        "field_minsim is allowed to leak near-misses — that leak IS the research problem. Safety check is unrelated leak, which must stay near 0.",
        "hybrid_and (zero out pairs that share a proper noun AND have conflicting relation/category buckets) is a non-geometric signal. If it lifts AUROC without inflating unrelated, the Naima-class confound is entity-level, not embeddable.",
        "Leave-one-theme-out adapter: a win on the training themes that vanishes on the held-out theme is overfitting, not a product lever.",
        "Weighted reinforce scales α by how far doc-doc cosine sits above field_minsim. It can manufacture separation at the Hebbian layer even when cosine overlap remains. Unrelated must stay low or the scheme is cheating.",
    ]

    OUT_JSON.write_text(json.dumps(json_ready(results), indent=2), encoding="utf-8")
    log(f"Wrote {OUT_JSON}")
    write_report(results)
    log("DONE")


if __name__ == "__main__":
    main()
