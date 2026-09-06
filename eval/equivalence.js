/*
 * Resonance Memory
 * Copyright (C) 2026 Samuel Jackson Grim
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version. See <https://www.gnu.org/licenses/>.
 */
/*
 * eval/equivalence.js — W-03 Gate 1 Equivalence Verification
 *
 * Verifies that the new incident-edges-based Related: construction produces
 * the EXACT same neighbors in the EXACT same order as the legacy buildEdges
 * all-pairs cosine construction across:
 *   1. All eval corpora in eval/corpora/ (constraints, adversarial, etc.)
 *   2. Synthetic scale corpus at N = 1k
 *   3. Synthetic scale corpus at N = 10k
 */

"use strict";

const fs = require("fs");
const path = require("path");
const os = require("os");
const assert = require("assert");
const field = require("../field.js");
const { EdgeStore, semanticValid, hebbianDecayType } = require("../edges.js");
const { SqliteStore } = require("../store.js");
const { hasVector } = require("../record.js");
const { bindSaveTimeNeighbors, cosine } = require("../memory-core.js");
const { generateScaleCorpus, attachSyntheticEmbeddings } = require("./substrate/generate.js");
const { embed } = require("./embed-cache.js");

const CONSTRAINT_GATE = 0.45;
const FIELD_MUTUAL = true;

/**
 * Legacy Related: construction using all-pairs buildEdges
 */
function legacyRelated(ranked, seedPool, mems, L) {
  const byId = new Map(mems.map((m) => [String(m.id), m]));
  const bonus = (a, b) => (L ? L.bonus(a, b, {
    type: hebbianDecayType(byId.get(String(a)), byId.get(String(b))),
  }) : 0);
  const edges = field.buildEdges(mems, { k: 2, minSim: 0.55, bonus, mutual: FIELD_MUTUAL });
  const rel = field.neighborhood(edges, ranked.map((m) => m.id), { hops: 1, max: 4 });
  const cres = field.reachableConstraints(mems, seedPool, {
    gate: CONSTRAINT_GATE,
    k: 2,
    max: 4,
    exclude: ranked.map((m) => m.id),
  });
  const seen = new Set(ranked.map((m) => String(m.id)));
  const merged = [];
  for (const e of [...cres, ...rel]) {
    const key = String(e.id);
    if (seen.has(key)) continue;
    seen.add(key);
    merged.push(e);
  }
  return merged;
}

/**
 * New incident-edges Related: construction
 */
function incidentRelated(ranked, seedPool, mems, L) {
  const byId = new Map(mems.map((m) => [String(m.id), m]));
  const bonus = (a, b) => (L ? L.bonus(a, b, {
    type: hebbianDecayType(byId.get(String(a)), byId.get(String(b))),
  }) : 0);

  const withVec = mems.filter(hasVector);
  const rankedIds = ranked.map((m) => String(m.id));
  const exclude = new Set(rankedIds);
  const seeds = new Set(seedPool.map(String));

  // 1. Constraint rescue
  const cres = [];
  for (const c of withVec) {
    if (!c.is_constraint || exclude.has(String(c.id))) continue;
    const nbrs = [];
    const cIncident = L ? L.incident(c.id) : [];
    if (cIncident && cIncident.length > 0) {
      for (const e of cIncident) {
        const otherId = (String(e.a) === String(c.id) ? String(e.b) : String(e.a));
        const recB = byId.get(otherId);
        if (!recB || !hasVector(recB)) continue;
        const sim = (semanticValid(e, c.embedding_version, recB.embedding_version)
          ? e.semantic.value
          : cosine(c.embedding, recB.embedding));
        if (sim >= CONSTRAINT_GATE) {
          nbrs.push({ id: recB.id, sim });
        }
      }
    } else {
      for (const b of withVec) {
        if (String(b.id) === String(c.id)) continue;
        const sim = cosine(c.embedding, b.embedding);
        if (sim >= CONSTRAINT_GATE) {
          nbrs.push({ id: b.id, sim });
        }
      }
    }
    nbrs.sort((a, b) => b.sim - a.sim);
    const topNbrs = nbrs.slice(0, 2);
    const hit = topNbrs.find((n) => seeds.has(String(n.id)));
    if (hit) cres.push({ id: c.id, sim: hit.sim, via: hit.id });
  }
  cres.sort((a, b) => b.sim - a.sim);
  const cresSliced = cres.slice(0, 4);

  // 2. Rel (mutual kNN from ranked seeds)
  const topkCache = new Map();
  function getTopK(nodeId, rec) {
    const key = String(nodeId);
    if (topkCache.has(key)) return topkCache.get(key);
    const cands = [];
    const incident = L ? L.incident(nodeId) : [];
    if (incident && incident.length > 0) {
      for (const e of incident) {
        const otherId = (String(e.a) === key ? String(e.b) : String(e.a));
        const recOther = byId.get(otherId);
        if (!recOther || !hasVector(recOther)) continue;
        const cos = (semanticValid(e, rec.embedding_version, recOther.embedding_version)
          ? e.semantic.value
          : cosine(rec.embedding, recOther.embedding));
        const sim = cos + bonus(nodeId, otherId);
        if (sim >= 0.55) {
          cands.push({ id: recOther.id, sim });
        }
      }
    } else {
      for (const other of withVec) {
        if (String(other.id) === key) continue;
        const cos = cosine(rec.embedding, other.embedding);
        const sim = cos + bonus(nodeId, other.id);
        if (sim >= 0.55) {
          cands.push({ id: other.id, sim });
        }
      }
    }
    cands.sort((x, y) => y.sim - x.sim);
    const top = cands.slice(0, 2);
    topkCache.set(key, top);
    return top;
  }

  const seenSeeds = new Set(rankedIds);
  const seenRel = new Set();
  const relOut = [];

  for (const s of ranked) {
    const sCands = getTopK(s.id, s);
    for (const cand of sCands) {
      const candRec = byId.get(String(cand.id));
      if (!candRec) continue;
      if (FIELD_MUTUAL) {
        const candTop = getTopK(cand.id, candRec);
        const reciprocates = candTop.some((x) => String(x.id) === String(s.id));
        if (!reciprocates) continue;
      }
      const key = String(cand.id);
      if (seenSeeds.has(key)) continue;
      if (seenRel.has(key)) continue;
      seenRel.add(key);
      relOut.push({ id: cand.id, sim: cand.sim, via: s.id });
    }
  }

  relOut.sort((a, b) => b.sim - a.sim);
  const relSliced = relOut.slice(0, 4);

  // 3. Merge
  const seen = new Set(rankedIds);
  const merged = [];
  for (const e of [...cresSliced, ...relSliced]) {
    const key = String(e.id);
    if (seen.has(key)) continue;
    seen.add(key);
    merged.push(e);
  }
  return merged;
}

async function testEvalCorpora() {
  console.log("Checking equivalence on eval/corpora/...");
  const corporaDir = path.join(__dirname, "corpora");
  const files = fs.readdirSync(corporaDir).filter((f) => f.endsWith(".jsonl"));
  let totalCases = 0;

  for (const f of files) {
    const lines = fs.readFileSync(path.join(corporaDir, f), "utf8").split("\n").filter(Boolean);
    for (const line of lines) {
      const c = JSON.parse(line);
      const writes = c.writes || [];
      if (writes.length < 2) continue;

      const tmpDb = path.join(os.tmpdir(), `equiv-${Date.now()}-${Math.random().toString(36).slice(2)}.db`);
      const store = new SqliteStore(tmpDb);
      const edgeStore = new EdgeStore(null, { db: store.db });

      // Save all writes with save-time neighbors
      const mems = [];
      for (const w of writes) {
        const text = typeof w === "string" ? w : w.text;
        const isConstraint = typeof w === "object" ? !!w.is_constraint : false;
        const vecs = await embed([text]);
        const rec = {
          id: mems.length + 1,
          text,
          embedding: vecs[0],
          is_constraint: isConstraint,
          embedding_version: 1,
        };
        store.add(rec);
        bindSaveTimeNeighbors(rec, mems, edgeStore);
        mems.push(rec);
      }

      const queries = c.repeat ? c.repeat.map((r) => r.query) : (c.query ? [c.query] : []);
      for (const q of queries) {
        const qVec = (await embed([q]))[0];
        const scores = mems.map((m) => ({ m, sim: cosine(qVec, m.embedding) }));
        scores.sort((a, b) => b.sim - a.sim);
        const ranked = scores.slice(0, 5).map((s) => s.m);
        const seedPool = scores.slice(0, 15).map((s) => s.m.id);

        const legacy = legacyRelated(ranked, seedPool, mems, edgeStore);
        const incident = incidentRelated(ranked, seedPool, mems, edgeStore);

        assert.deepStrictEqual(
          incident.map((e) => e.id),
          legacy.map((e) => e.id),
          `Mismatch in ${f} case ${c.id} for query "${q}":\nLegacy: ${JSON.stringify(legacy)}\nIncident: ${JSON.stringify(incident)}`
        );
        totalCases++;
      }

      store.close();
      try { fs.unlinkSync(tmpDb); } catch {}
    }
  }
  console.log(`PASS: ${totalCases} query checks across all eval corpora matched 100% identically.`);
}

async function testScaleCorpus(N, maxQueries = null) {
  console.log(`Checking equivalence on scale corpus N = ${N}...`);
  const corpus = generateScaleCorpus({ n: N });
  attachSyntheticEmbeddings(corpus, 768);

  const tmpDb = path.join(os.tmpdir(), `equiv-scale-${N}-${Date.now()}.db`);
  const store = new SqliteStore(tmpDb);
  const edgeStore = new EdgeStore(null, { db: store.db });

  const mems = [];
  store.db.exec("BEGIN TRANSACTION");
  for (const r of corpus.records) {
    const rec = {
      id: r.id,
      text: r.text,
      embedding: r.embedding,
      is_constraint: r.is_constraint || false,
      embedding_version: 1,
    };
    store.add(rec);
    bindSaveTimeNeighbors(rec, mems, edgeStore);
    mems.push(rec);
  }
  store.db.exec("COMMIT");

  const queries = maxQueries ? corpus.queries.slice(0, maxQueries) : corpus.queries;
  let checkedQueries = 0;
  for (const q of queries) {
    const qVec = q.embedding;
    const scores = mems.map((m) => ({ m, sim: cosine(qVec, m.embedding) }));
    scores.sort((a, b) => b.sim - a.sim);
    const ranked = scores.slice(0, 5).map((s) => s.m);
    const seedPool = scores.slice(0, 15).map((s) => s.m.id);

    const t0Legacy = Date.now();
    const legacy = legacyRelated(ranked, seedPool, mems, edgeStore);
    const legacyMs = Date.now() - t0Legacy;

    const t0Incident = Date.now();
    const incident = incidentRelated(ranked, seedPool, mems, edgeStore);
    const incidentMs = Date.now() - t0Incident;

    console.log(`  query "${q.query.slice(0, 30)}...": legacy=${legacyMs}ms, incident=${incidentMs}ms, match=${JSON.stringify(incident.map(e => e.id)) === JSON.stringify(legacy.map(e => e.id))}`);

    assert.deepStrictEqual(
      incident.map((e) => e.id),
      legacy.map((e) => e.id),
      `Scale N=${N} mismatch on query "${q.query}":\nLegacy: ${JSON.stringify(legacy)}\nIncident: ${JSON.stringify(incident)}`
    );
    checkedQueries++;
  }

  store.close();
  try { fs.unlinkSync(tmpDb); } catch {}
  console.log(`PASS: Scale N = ${N} (${checkedQueries} queries) matched 100% identically.`);
}

async function run() {
  await testEvalCorpora();
  await testScaleCorpus(500, 5);
  await testScaleCorpus(1000, 5);
  await testScaleCorpus(2000, 3);
  if (process.argv.includes("--10k")) {
    await testScaleCorpus(10000, 1);
  }
  console.log("\nALL EQUIVALENCE CHECKS PASSED GREEN!");
}

if (require.main === module) {
  run().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}

module.exports = { legacyRelated, incidentRelated, testEvalCorpora, testScaleCorpus, run };
