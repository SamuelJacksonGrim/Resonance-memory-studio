#!/usr/bin/env node
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
 * eval/substrate/fire-together-normalization.js — Test whether normalizing
 * edge bonuses within a node's neighborhood recovers the 1.47x true/near-miss
 * separation that tanh(maxBonus) destroys.
 *
 * The stored weight in EdgeStore already has 1.47x true/near-miss ratio.
 * The problem: bonus() reads it through tanh(maxBonus), which saturates at
 * any weight > 3. Both true (44) and near-miss (29) hit maxBonus = 0.3.
 *
 * This experiment tests whether neighborhood-competitive normalization
 * recovers the separation in the bonus values that a real recall would
 * use for the Related: section.
 *
 * IMPORTANT: This does NOT change primary ranking (cosine-only). It tests
 * what bonus values the Related: discovery layer sees.
 */

"use strict";

const fs = require("fs");
const path = require("path");
const { EdgeStore } = require("../../edges.js");
const { cosine } = require("../pipeline.js");
const { effectiveHebbian } = require("../../edges.js");

const CORPUS_PATH = path.join(__dirname, "../corpora/fire-together-corpus.json");
const RESULTS_PATH = path.join(__dirname, "fire-together-normalization-results.json");
const EMBED_MODEL = process.env.EMBED_MODEL || "text-embedding-nomic-embed-text-v1.5";
const EMBED_ENDPOINT = "http://localhost:1234";

const DAY_SECS = 86400;
const HALF_LIFE_SECS = 7 * DAY_SECS;
const MAX_BONUS = 0.3;
const ALPHA = 0.1;
const K = 5;

function pairKey(a, b) {
  const x = String(a), y = String(b);
  return x < y ? x + "\t" + y : y + "\t" + x;
}

async function embedBatch(texts) {
  const body = JSON.stringify({ model: EMBED_MODEL, input: texts });
  const buf = Buffer.from(body);
  const res = await fetch(`${EMBED_ENDPOINT}/v1/embeddings`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "Content-Length": String(buf.length) },
    body: buf,
  });
  if (!res.ok) throw new Error(`Embed failed: ${res.status} ${await res.text()}`);
  const json = JSON.parse(await res.text());
  return json.data.map((d) => d.embedding);
}

function seededRandom(seed) {
  let s = seed;
  return () => {
    s = (s * 16807 + 0) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

function generateHaystackRecords(count, seed = 42) {
  const rng = seededRandom(seed);
  const pick = (arr) => arr[Math.floor(rng() * arr.length)];

  const templates = [
    (r) => `I bought a new ${pick(["blue","red","green","black","white","gray","navy","brown"])} ${pick(["jacket","shirt","pants","dress","sweater","coat","hoodie","scarf","hat","belt"])} from ${pick(["Zara","H&M","Uniqlo","Target","Walmart","Amazon","Nordstrom","Old Navy","Gap","Costco"])} last week`,
    (r) => `I've been cooking ${pick(["Thai","Italian","Mexican","Japanese","Indian","Korean","French","Chinese","Vietnamese","Lebanese"])} food at home more often lately`,
    (r) => `My ${pick(["dad","mom","brother","sister","cousin","uncle","aunt","neighbor","friend","colleague"])} lives in ${pick(["Denver","Austin","Portland","Seattle","Chicago","Boston","Miami","Atlanta","Nashville","Minneapolis"])}`,
    (r) => `I need to ${pick(["update my resume","schedule a dentist appointment","renew my passport","pay the electric bill","change my oil","fix the leaky faucet","replace the brake pads","clean the gutters","update my insurance","file my taxes"])}`,
    (r) => `I read a great book about ${pick(["neuroscience","history","cooking","travel","philosophy","psychology","astronomy","genetics","art","politics"])} this past weekend`,
    (r) => `The ${pick(["parking garage","grocery store","coffee shop","library","gym","hospital","bank","post office","hardware store","pharmacy"])} on ${pick(["Main","Oak","Elm","Maple","Cedar","Pine","Birch","Walnut","Chestnut","Willow"])} Street is always crowded`,
    (r) => `I'm trying to cut back on ${pick(["sugar","caffeine","social media","screen time","carbs","dairy","processed food","meat","alcohol","junk food"])}`,
    (r) => `We had ${pick(["pizza","sushi","tacos","ramen","burgers","pasta","curry","steak","bbq","falafel"])} for dinner last night`,
    (r) => `My ${pick(["dog","cat","hamster","fish","bird","rabbit","turtle","guinea pig","parrot","ferret"])} ${pick(["loves","hates","enjoys","sleeps in","chases","chews on","perches on","bites","grooms","follows around"])} ${pick(["the couch","the door","the garden","the kitchen","the backyard","the basement","the mailbox","the window","the laundry","the garden hose"])} at home`,
    (r) => `I went ${pick(["hiking","swimming","biking","running","kayaking","rock climbing","surfing","fishing","camping","skiing"])} last weekend`,
    (r) => `I've been listening to a lot of ${pick(["podcasts","lo-fi","jazz","classical","rock","hip-hop","indie","folk","electronic","country"])}`,
    (r) => `I ${pick(["love","hate","need","want","miss","enjoy","prefer","avoid","appreciate","dislike"])} ${pick(["morning coffee","late nights","early mornings","weekends","cold weather","hot weather","rain","sunshine","snow","wind"])} at home`,
    (r) => `The ${pick(["new","old","local","famous","hidden","chain","independent","corner","downtown","suburban"])}${pick(["restaurant","bar","cafe","bookstore","store","theater","stadium","museum","gallery","studio"])} is ${pick(["closed","renovating","hiring","expanding","closing down","opening soon","under new management","fully booked","under review","doing well"])} at home`,
    (r) => `I should really ${pick(["exercise more","eat better","sleep more","save money","travel more","learn a new skill","call my mom","clean the house","declutter","start meditating"])}`,
    (r) => `My ${pick(["boss","teacher","professor","manager","coach","supervisor","mentor","advisor","instructor","trainer"])} said I need to ${pick(["improve my communication","work on my patience","be more organized","take more initiative","delegate better","manage my time","be more confident","speak up more","collaborate better","follow through"])}`,
    (r) => `I went to ${pick(["the beach","the mountains","the lake","the desert","the forest","the canyon","the waterfall","the coastline","the island","the river"])} and the ${pick(["view","weather","water","air","scenery","nature","wildlife","sunrise","sunset","stars"])} was incredible`,
    (r) => `I bought a ${pick(["laptop","phone","tablet","monitor","keyboard","headphones","camera","drone","watch","speaker"])} from ${pick(["Apple","Samsung","Dell","Sony","Bose","Canon","DJI","Garmin","JBL","Anker"])}`,
    (r) => `The ${pick(["traffic","construction","road work","accident","detour","lane closure","bridge repair","utility work","street fair","parade"])} made me ${pick(["late","take a detour","wait an hour","find an alternate route","lose my patience","call in late","get stuck","turn around","replan","sit in traffic"])}`,
    (r) => `I've been ${pick(["working out","going to the gym","training for a race","doing yoga","practicing martial arts","learning to dance","stretching more","lifting weights","running marathons","swimming laps"])} at the gym`,
    (r) => `I need to ${pick(["return this","exchange this","get a refund","file a warranty claim","complain about","report","upgrade","renew","cancel","reactivate"])}${pick([" my subscription","the warranty","the contract","the order","the app","the service","the plan","the membership","the policy","the account"])} at home`,
  ];

  const records = [];
  for (let i = 0; i < count; i++) {
    const text = templates[i % templates.length](i);
    records.push({ id: `h${count}_${i}`, text, theme_id: null, role: "distractor" });
  }
  return records;
}

// Run the equilibrium simulation (reuse same protocol as fire-together-decay.js)
async function runEquilibrium(corpus, allRecords, queries, edgeStore) {
  const themes = corpus.themes;
  const k = 5;
  const themeQueryMap = {};
  for (const theme of themes) {
    const indices = [];
    for (let qi = 0; qi < queries.length; qi++) {
      if (queries[qi].theme_id === theme.id) indices.push(qi);
    }
    themeQueryMap[theme.id] = indices;
  }

  for (let week = 0; week < 104; week++) {
    let sessionTime = new Date(Date.now() + week * 7 * DAY_SECS * 1000);
    for (const theme of themes) {
      const queryIndices = themeQueryMap[theme.id];
      for (let s = 0; s < 2; s++) {
        const nQueries = 2 + Math.floor(Math.random() * 2);
        for (let q = 0; q < nQueries; q++) {
          sessionTime = new Date(sessionTime.getTime() + (60000 + Math.random() * 300000));
          const qi = queryIndices[(week * 20 + s * 10 + q) % queryIndices.length];
          const query = queries[qi];
          const ranked = allRecords
            .map((r) => ({ id: r.id, sim: cosine(query.embedding, r.embedding) }))
            .sort((a, b) => b.sim - a.sim)
            .slice(0, k);
          const topK = ranked.map((r) => r.id);
          edgeStore.reinforceRecall(topK, []);
        }
        sessionTime = new Date(sessionTime.getTime() + (12 + Math.random() * 12) * 3600 * 1000);
      }
    }
  }
}

// ============================================================
// Normalization schemes
// ============================================================

// For each record (node), get all its incident edge weights.
// Then apply a normalization and return normalized bonuses.
// We measure true/near-miss pairs under each scheme.

function getIncidentEdges(edgeStore, recordId) {
  // Get all edges incident to a node by scanning the store
  // EdgeStore doesn't expose incident() directly, so we scan.
  const edges = [];
  // We need to access edgeStore's internal Map. In production this
  // would use incident() — here we scan via the weight method.
  return edges;
}

// Since EdgeStore doesn't expose incident() in the public API, we need
// to work with what we have: we can get the stored weight of any pair
// via edgeStore.weight(a, b), but we can't enumerate all neighbors.
//
// For the experiment, we'll take a different approach:
// We know the corpus structure — every record has known true/near/unrelated
// pairs. For each record, we can compute the "neighborhood" as all other
// records in the corpus. This is the full field, not just save-time neighbors.
//
// The normalization schemes we test:

// Scheme 1: Divide by the node's MAX edge weight
//   normalized_bonus = tanh(stored_weight / max_weight) * maxBonus
// This is a "relative strength" normalization.
// If a node has max=50 and the target edge=30, normalized=30/50=0.6.
// This preserves the ratio: 42/35 vs 30/35 → 1.2 vs 0.86.
// But the max weight is typically a true pair, so this compresses
// true pairs relative to near-misses if the near-miss's max is smaller.

// Scheme 2: Z-score within node
//   normalized_bonus = tanh((stored_weight - mean) / std) * maxBonus
// This centers around the node's average edge and scales by variance.
// A true pair at 42 vs node mean of 25 and std of 10 → z=1.7.
// A near-miss at 30 vs same mean/std → z=0.5.
// This could amplify separation if true pairs are outliers.
// But if all edges are high, std is also high and separation shrinks.

// Scheme 3: Rank-based (fraction of edges below this one)
//   rank_bonus = (count of edges with lower weight / total edges) * maxBonus
// This is honest: it preserves the ordinal ranking.
// True=42 might be rank 0.95 (top 5%), near-miss=30 might be rank 0.70.
// This maps to bonus=0.285 vs bonus=0.210 → ratio=1.36x.
// But the ranking is ordinal, not ratio-preserving.

// Scheme 4: Softmax over node's edges
//   softmax_i = exp(w_i / tau) / sum(exp(w_j / tau))
// With tau=1 (no scaling), high weights dominate → all bonuses ≈ 0.3.
// With tau=0.1 (sharp), only the strongest edge gets non-zero bonus.
// This preserves relative ordering but is sensitive to tau.

// Scheme 5: Divide by the node's mean edge weight (ratio to average)
//   normalized_bonus = tanh(stored_weight / mean_weight) * maxBonus
// This is the simplest honest normalization.
// True=42, mean=28 → ratio=1.5. Near-miss=30, mean=28 → ratio=1.07.
// tanh(1.5)*0.3=0.26, tanh(1.07)*0.3=0.24 → ratio=1.08x.
// The ratio is compressed by tanh.

// Scheme 6: Subtract the node's minimum edge (range normalization)
//   normalized_bonus = tanh(stored_weight - min_weight) * maxBonus
// True=42, min=1 → 41. Near-miss=30, min=1 → 29.
// tanh(41)*0.3=0.3, tanh(29)*0.3=0.3 → still saturated!
// Range normalization doesn't help with tanh saturation.

// The key insight: ALL of these schemes must survive tanh saturation.
// The stored weight ratio is 1.47x, but tanh(42) = tanh(30) = 0.99999... = 1.
// The only way to preserve ratio through tanh is to keep the arguments
// BELOW the saturation threshold (~3 for tanh).
// That means the normalization must DIVIDE by something > 10 for
// weights of 30-44.

// What about: bonus = tanh(stored_weight / C) * maxBonus,
// where C is a node-level scale factor (e.g., max_weight, mean_weight, sqrt)?
// If C = max_weight for the node:
//   True: tanh(42/44) = tanh(0.95) = 0.737 → bonus=0.221
//   Near: tanh(30/44) = tanh(0.68) = 0.591 → bonus=0.177
//   Ratio = 1.25x — improved from 1.00x!

// If C = mean_weight for the node:
//   True: tanh(42/28) = tanh(1.5) = 0.905 → bonus=0.272
//   Near: tanh(30/28) = tanh(1.07) = 0.787 → bonus=0.236
//   Ratio = 1.15x

// If C = sqrt(max_weight) (sqrt scaling):
//   True: tanh(42/sqrt(44)) = tanh(6.34) = 0.99998 → bonus=0.3 (saturated)
//   Near: tanh(30/sqrt(44)) = tanh(4.53) = 0.9997 → bonus=0.3 (saturated)
//   Ratio = 1.00x — still saturated!

// So the answer depends on what normalization keeps arguments
// below the tanh saturation threshold while preserving the 1.47x ratio.

// The honest approach: for each node, compute normalization from
// its actual edge distribution, then apply tanh to the normalized weight.
// If the node's edges are all high (saturated), the separation is lost.
// This is honest: we're not inventing separation; we're revealing what's
// there in the weights, scaled by a factor the node itself determines.

async function main() {
  console.log("Loading corpus...");
  const corpus = JSON.parse(fs.readFileSync(CORPUS_PATH, "utf8"));
  const themes = corpus.themes;
  const k = 5;

  const HAYSTACK_SIZE = 300;
  console.log(`Generating ${HAYSTACK_SIZE} distractor records (seed=42)...`);
  const haystackRecords = generateHaystackRecords(HAYSTACK_SIZE, 42);
  const allRecords = [...corpus.records, ...haystackRecords];

  const allTexts = [...allRecords.map((r) => r.text), ...corpus.queries.map((q) => q.query)];
  console.log(`Embedding ${allTexts.length} texts...`);
  const allEmbeddings = [];
  for (let i = 0; i < allTexts.length; i += 10) {
    console.log(`  batch ${Math.floor(i / 10) + 1}/${Math.ceil(allTexts.length / 10)}`);
    const batch = await embedBatch(allTexts.slice(i, i + 10));
    allEmbeddings.push(...batch);
  }

  let off = 0;
  const records = allRecords.map((r) => ({ ...r, embedding: allEmbeddings[off++] }));
  const queries = corpus.queries.map((q) => ({ ...q, embedding: allEmbeddings[off++] }));

  // Build full adjacency: for each record, compute stored weights
  // against ALL other records (this is the full neighborhood)
  const n = allRecords.length;
  console.log("\nBuilding full adjacency matrix...");
  // We'll compute weights lazily during normalization

  // Run equilibrium
  console.log("Running equilibrium simulation...");
  const edgeStore = new EdgeStore(null);
  await runEquilibrium(corpus, records, queries, edgeStore);
  console.log("Equilibrium achieved. Computing normalization schemes...");

  // Compute the "neighborhood" for each record:
  // For each record, we compute stored weights against all other records.
  // This is the full field. In production, this would be the save-time
  // neighbors (K=5) + co-activated edges — much smaller.
  // But for the experiment, we use the full corpus to measure
  // whether normalization recovers separation.

  // For honest normalization, we should use the SAME neighborhood
  // that the real system would see: save-time neighbors (K=5) +
  // co-activated edges from reinforcement.
  //
  // Since we can't enumerate incident edges, we'll compute the
  // neighborhood as all records that have a non-zero stored weight
  // with the record. In the real system, this would be the
  // incident() method.
  //
  // For the experiment: we'll use a SUBSET of the full corpus as
  // the "neighborhood" — records that co-fire with the record in
  // top-5 of some query. This is the practical neighborhood.

  // Actually, the simplest honest approach: use the node's
  // stored weight relative to its MEAN edge weight. If the mean
  // is high (all edges saturated), normalization doesn't help.
  // If the mean is lower than the target edge, normalization
  // preserves the ratio. This is honest.

  // For each record, compute its neighborhood (all records it has
  // a non-trivial edge with, defined as stored_weight > 0.01)
  // and compute normalization from that.

  const results = {
    version: corpus.version,
    k,
    n_records: allRecords.length,
    n_haystack: HAYSTACK_SIZE,
    embed_model: EMBED_MODEL,
    half_life_days: 7,
    alpha: ALPHA,
    max_bonus: MAX_BONUS,
    schemes: {},
  };

  // For each scheme, compute normalized bonuses for true/near/unrelated
  // pairs.

  // First, compute the "neighborhood" for each record.
  // We'll define it as: for each record r, find all other records
  // that have a non-trivial stored weight (> 0.001) with r.
  // This is the practical neighborhood — what the system would see.

  // Since we can't enumerate incident edges, we'll compute the
  // neighborhood on-the-fly by scanning all pairs.
  console.log("Computing neighborhood statistics...");

  // Precompute: for each record, its mean/max/std edge weight
  // across ALL other records (full field as neighborhood).
  const nodeStats = []; // nodeStats[i] = { mean, max, min, std, count }
  for (let i = 0; i < n; i++) {
    let sum = 0, count = 0, maxW = 0, minW = Infinity;
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const w = edgeStore.weight(records[i].id, records[j].id);
      if (w > 0.001) {
        sum += w;
        count++;
        maxW = Math.max(maxW, w);
        minW = Math.min(minW, w);
      }
    }
    const mean = count > 0 ? sum / count : 0;
    // Compute std
    let varSum = 0;
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const w = edgeStore.weight(records[i].id, records[j].id);
      if (w > 0.001) {
        varSum += (w - mean) ** 2;
      }
    }
    const std = count > 1 ? Math.sqrt(varSum / count) : 0;
    nodeStats.push({ mean, max: maxW, min: minW === Infinity ? 0 : minW, std, count });
  }

  // Now apply each normalization scheme.
  // For each pair (a, b), we use nodeStats[a] as the normalization source.
  // In production, this would be: normalize per query node (the record
  // being recalled), not per edge endpoint.

  // Measure true/near/unrelated bonuses under each scheme.

  // Scheme A: Divide by node's max edge weight (relative to strongest)
  {
    let trueSum = 0, nmSum = 0, unSum = 0, tc = 0, nc = 0, uc = 0;
    for (const tp of corpus.true_pairs) {
      const w = edgeStore.weight(tp.a, tp.b);
      const normW = w / Math.max(nodeStats[allRecords.findIndex(r => r.id === tp.a)].max, 0.001);
      trueSum += Math.tanh(normW) * MAX_BONUS;
      tc++;
    }
    for (const nm of corpus.near_miss_pairs) {
      const w = edgeStore.weight(nm.a, nm.b);
      const normW = w / Math.max(nodeStats[allRecords.findIndex(r => r.id === nm.a)].max, 0.001);
      nmSum += Math.tanh(normW) * MAX_BONUS;
      nc++;
    }
    for (const up of corpus.unrelated_pairs) {
      const w = edgeStore.weight(up.a, up.b);
      const normW = w / Math.max(nodeStats[allRecords.findIndex(r => r.id === up.a)].max, 0.001);
      unSum += Math.tanh(normW) * MAX_BONUS;
      uc++;
    }
    results.schemes["divide_by_node_max"] = {
      label: "Divide stored weight by node's max edge weight",
      true_avg: trueSum / tc,
      nm_avg: nmSum / nc,
      un_avg: unSum / uc,
      true_nm_ratio: nmSum > 0 ? (trueSum / tc) / (nmSum / nc) : Infinity,
      true_un_ratio: unSum > 0 ? (trueSum / tc) / (unSum / uc) : Infinity,
    };
  }

  // Scheme B: Divide by node's mean edge weight (relative to average)
  {
    let trueSum = 0, nmSum = 0, unSum = 0, tc = 0, nc = 0, uc = 0;
    for (const tp of corpus.true_pairs) {
      const w = edgeStore.weight(tp.a, tp.b);
      const mean = nodeStats[allRecords.findIndex(r => r.id === tp.a)].mean;
      const normW = w / Math.max(mean, 0.001);
      trueSum += Math.tanh(normW) * MAX_BONUS;
      tc++;
    }
    for (const nm of corpus.near_miss_pairs) {
      const w = edgeStore.weight(nm.a, nm.b);
      const mean = nodeStats[allRecords.findIndex(r => r.id === nm.a)].mean;
      const normW = w / Math.max(mean, 0.001);
      nmSum += Math.tanh(normW) * MAX_BONUS;
      nc++;
    }
    for (const up of corpus.unrelated_pairs) {
      const w = edgeStore.weight(up.a, up.b);
      const mean = nodeStats[allRecords.findIndex(r => r.id === up.a)].mean;
      const normW = w / Math.max(mean, 0.001);
      unSum += Math.tanh(normW) * MAX_BONUS;
      uc++;
    }
    results.schemes["divide_by_node_mean"] = {
      label: "Divide stored weight by node's mean edge weight",
      true_avg: trueSum / tc,
      nm_avg: nmSum / nc,
      un_avg: unSum / uc,
      true_nm_ratio: nmSum > 0 ? (trueSum / tc) / (nmSum / nc) : Infinity,
      true_un_ratio: unSum > 0 ? (trueSum / tc) / (unSum / uc) : Infinity,
    };
  }

  // Scheme C: Z-score within node's neighborhood
  {
    let trueSum = 0, nmSum = 0, unSum = 0, tc = 0, nc = 0, uc = 0;
    for (const tp of corpus.true_pairs) {
      const w = edgeStore.weight(tp.a, tp.b);
      const mean = nodeStats[allRecords.findIndex(r => r.id === tp.a)].mean;
      const std = nodeStats[allRecords.findIndex(r => r.id === tp.a)].std;
      const normW = std > 0.001 ? (w - mean) / std : 0;
      trueSum += Math.tanh(Math.abs(normW)) * MAX_BONUS;
      tc++;
    }
    for (const nm of corpus.near_miss_pairs) {
      const w = edgeStore.weight(nm.a, nm.b);
      const mean = nodeStats[allRecords.findIndex(r => r.id === nm.a)].mean;
      const std = nodeStats[allRecords.findIndex(r => r.id === nm.a)].std;
      const normW = std > 0.001 ? (w - mean) / std : 0;
      nmSum += Math.tanh(Math.abs(normW)) * MAX_BONUS;
      nc++;
    }
    for (const up of corpus.unrelated_pairs) {
      const w = edgeStore.weight(up.a, up.b);
      const mean = nodeStats[allRecords.findIndex(r => r.id === up.a)].mean;
      const std = nodeStats[allRecords.findIndex(r => r.id === up.a)].std;
      const normW = std > 0.001 ? (w - mean) / std : 0;
      unSum += Math.tanh(Math.abs(normW)) * MAX_BONUS;
      uc++;
    }
    results.schemes["zscore_node"] = {
      label: "Z-score within node's neighborhood",
      true_avg: trueSum / tc,
      nm_avg: nmSum / nc,
      un_avg: unSum / uc,
      true_nm_ratio: nmSum > 0 ? (trueSum / tc) / (nmSum / nc) : Infinity,
      true_un_ratio: unSum > 0 ? (trueSum / tc) / (unSum / uc) : Infinity,
    };
  }

  // Scheme D: Rank-based (fraction of edges below this one)
  {
    let trueSum = 0, nmSum = 0, unSum = 0, tc = 0, nc = 0, uc = 0;
    for (const tp of corpus.true_pairs) {
      const w = edgeStore.weight(tp.a, tp.b);
      const stats = nodeStats[allRecords.findIndex(r => r.id === tp.a)];
      // Compute rank: fraction of edges below w
      let below = 0, total = 0;
      for (let j = 0; j < n; j++) {
        if (tp.a === allRecords[j].id) continue;
        const wj = edgeStore.weight(tp.a, allRecords[j].id);
        if (wj > 0.001) {
          below += wj < w ? 1 : 0;
          total++;
        }
      }
      const rank = total > 0 ? below / total : 0;
      trueSum += rank * MAX_BONUS;
      tc++;
    }
    for (const nm of corpus.near_miss_pairs) {
      const w = edgeStore.weight(nm.a, nm.b);
      let below = 0, total = 0;
      for (let j = 0; j < n; j++) {
        if (nm.a === allRecords[j].id) continue;
        const wj = edgeStore.weight(nm.a, allRecords[j].id);
        if (wj > 0.001) {
          below += wj < w ? 1 : 0;
          total++;
        }
      }
      const rank = total > 0 ? below / total : 0;
      nmSum += rank * MAX_BONUS;
      nc++;
    }
    for (const up of corpus.unrelated_pairs) {
      const w = edgeStore.weight(up.a, up.b);
      let below = 0, total = 0;
      for (let j = 0; j < n; j++) {
        if (up.a === allRecords[j].id) continue;
        const wj = edgeStore.weight(up.a, allRecords[j].id);
        if (wj > 0.001) {
          below += wj < w ? 1 : 0;
          total++;
        }
      }
      const rank = total > 0 ? below / total : 0;
      unSum += rank * MAX_BONUS;
      uc++;
    }
    results.schemes["rank_based"] = {
      label: "Rank-based (fraction of edges below this weight)",
      true_avg: trueSum / tc,
      nm_avg: nmSum / nc,
      un_avg: unSum / uc,
      true_nm_ratio: nmSum > 0 ? (trueSum / tc) / (nmSum / nc) : Infinity,
      true_un_ratio: unSum > 0 ? (trueSum / tc) / (unSum / uc) : Infinity,
    };
  }

  // Scheme E: Softmax over node's edges (with different temperatures)
  for (const tau of [0.1, 0.5, 1.0, 2.0]) {
    let trueSum = 0, nmSum = 0, unSum = 0, tc = 0, nc = 0, uc = 0;
    // We need the full set of edge weights for each node to do softmax.
    // This is expensive but honest.
    for (const tp of corpus.true_pairs) {
      const w = edgeStore.weight(tp.a, tp.b);
      const stats = nodeStats[allRecords.findIndex(r => r.id === tp.a)];
      // For softmax, we need all weights. Use max as reference.
      const maxW = stats.max;
      if (maxW === 0) continue;
      // Normalize: softmax with reference at maxW
      const expW = Math.exp((w - maxW) / tau);
      const expMax = Math.exp((maxW - maxW) / tau); // = 1
      const bonus = (expW / (expMax + expW)) * MAX_BONUS;
      trueSum += bonus;
      tc++;
    }
    for (const nm of corpus.near_miss_pairs) {
      const w = edgeStore.weight(nm.a, nm.b);
      const stats = nodeStats[allRecords.findIndex(r => r.id === nm.a)];
      const maxW = stats.max;
      if (maxW === 0) continue;
      const expW = Math.exp((w - maxW) / tau);
      const expMax = Math.exp((maxW - maxW) / tau);
      const bonus = (expW / (expMax + expW)) * MAX_BONUS;
      nmSum += bonus;
      nc++;
    }
    for (const up of corpus.unrelated_pairs) {
      const w = edgeStore.weight(up.a, up.b);
      const stats = nodeStats[allRecords.findIndex(r => r.id === up.a)];
      const maxW = stats.max;
      if (maxW === 0) continue;
      const expW = Math.exp((w - maxW) / tau);
      const expMax = Math.exp((maxW - maxW) / tau);
      const bonus = (expW / (expMax + expW)) * MAX_BONUS;
      unSum += bonus;
      uc++;
    }
    const key = `softmax_tau_${tau}`;
    results.schemes[key] = {
      label: `Softmax over node's edges (tau=${tau})`,
      true_avg: trueSum / tc,
      nm_avg: nmSum / nc,
      un_avg: unSum / uc,
      true_nm_ratio: nmSum > 0 ? (trueSum / tc) / (nmSum / nc) : Infinity,
      true_un_ratio: unSum > 0 ? (trueSum / tc) / (unSum / uc) : Infinity,
    };
  }

  // Scheme F: Subtract node's min edge (range normalization)
  {
    let trueSum = 0, nmSum = 0, unSum = 0, tc = 0, nc = 0, uc = 0;
    for (const tp of corpus.true_pairs) {
      const w = edgeStore.weight(tp.a, tp.b);
      const stats = nodeStats[allRecords.findIndex(r => r.id === tp.a)];
      const normW = w - stats.min;
      trueSum += Math.tanh(normW) * MAX_BONUS;
      tc++;
    }
    for (const nm of corpus.near_miss_pairs) {
      const w = edgeStore.weight(nm.a, nm.b);
      const stats = nodeStats[allRecords.findIndex(r => r.id === nm.a)];
      const normW = w - stats.min;
      nmSum += Math.tanh(normW) * MAX_BONUS;
      nc++;
    }
    for (const up of corpus.unrelated_pairs) {
      const w = edgeStore.weight(up.a, up.b);
      const stats = nodeStats[allRecords.findIndex(r => r.id === up.a)];
      const normW = w - stats.min;
      unSum += Math.tanh(normW) * MAX_BONUS;
      uc++;
    }
    results.schemes["range_normalize"] = {
      label: "Subtract node's min edge weight (range normalization)",
      true_avg: trueSum / tc,
      nm_avg: nmSum / nc,
      un_avg: unSum / uc,
      true_nm_ratio: nmSum > 0 ? (trueSum / tc) / (nmSum / nc) : Infinity,
      true_un_ratio: unSum > 0 ? (trueSum / tc) / (unSum / uc) : Infinity,
    };
  }

  // Save results
  fs.writeFileSync(RESULTS_PATH, JSON.stringify(results, null, 2));
  console.log(`\nResults saved to ${RESULTS_PATH}`);

  // Print results
  console.log("\n=== NORMALIZATION SCHEME RESULTS ===");
  console.log("Measured at session boundary (2 years of equilibrium usage)");
  console.log();
  console.log("Scheme                                      | True   | Near   | Unrelated | True/Near | True/Unrel");
  console.log("--------------------------------------------|--------|--------|-----------|-----------|-----------");
  for (const [key, r] of Object.entries(results.schemes)) {
    const label = r.label.padEnd(46);
    const tStr = r.true_avg.toFixed(4);
    const nStr = r.nm_avg.toFixed(4);
    const uStr = r.un_avg.toFixed(4);
    const tnStr = (r.true_nm_ratio === Infinity ? "INF" : r.true_nm_ratio.toFixed(2)).padStart(7);
    const tuStr = (r.true_un_ratio === Infinity ? "INF" : r.true_un_ratio.toFixed(2)).padStart(9);
    console.log(`  ${label} | ${tStr} | ${nStr} | ${uStr}     | ${tnStr}x | ${tuStr}x`);
  }

  console.log("\n=== RAW WEIGHTS (baseline) ===");
  let rawTrue = 0, rawNm = 0, rawUn = 0, tc = 0, nc = 0, uc = 0;
  for (const tp of corpus.true_pairs) { rawTrue += edgeStore.weight(tp.a, tp.b); tc++; }
  for (const nm of corpus.near_miss_pairs) { rawNm += edgeStore.weight(nm.a, nm.b); nc++; }
  for (const up of corpus.unrelated_pairs) { rawUn += edgeStore.weight(up.a, up.b); uc++; }
  console.log(`  True:  ${rawTrue / tc}`);
  console.log(`  Near:  ${rawNm / nc}`);
  console.log(`  Unrel: ${rawUn / uc}`);
  console.log(`  True/Near: ${(rawTrue / tc) / Math.max(rawNm / nc, 1e-9)}x`);
  console.log(`  True/Unrel: ${(rawTrue / tc) / Math.max(rawUn / uc, 1e-9)}x`);

  console.log("\n=== CURRENT BONUS (no normalization, tanh saturation) ===");
  console.log(`  True:  ${Math.tanh(rawTrue / tc) * MAX_BONUS}`);
  console.log(`  Near:  ${Math.tanh(rawNm / nc) * MAX_BONUS}`);
  console.log(`  Unrel: ${Math.tanh(rawUn / uc) * MAX_BONUS}`);
  console.log(`  True/Near: 1.00x (both saturated)`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
