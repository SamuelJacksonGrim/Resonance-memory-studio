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
 * eval/substrate/fire-together-reinforce.js — Measure Hebbian accumulation
 * under RM's actual bet: repeated co-activation over time sharpens true-pair
 * vs near-miss separation.
 *
 * Protocol:
 *   1. Embed everything with nomic-embed-text (deterministic haystack)
 *   2. For each theme, run scripted "sessions" (paraphrase queries)
 *   3. Each session: recall -> reinforceRecall(topK)
 *  4. After each session batch, measure the edgeStore bonus for:
 *      a) true pairs
 *      b) near-miss pairs
 *      c) unrelated pairs
 *   5. Also measure cosine-based cofire (top-k overlap) after each session
 *      batch — this is the "pre-hebbian" baseline
 *   6. Compare the convergence curves
 *
 * The key question: does the edgeStore bonus separate true pairs from
 * near-misses even when their cosine similarity is similar?
 *
 * Usage:
 *   node eval/substrate/fire-together-reinforce.js
 */

"use strict";

const fs = require("fs");
const path = require("path");
const { EdgeStore } = require("../../edges.js");
const { cosine } = require("../pipeline.js");

const CORPUS_PATH = path.join(__dirname, "../corpora/fire-together-corpus.json");
const RESULTS_PATH = path.join(__dirname, "fire-together-reinforce-results.json");
const EMBED_MODEL = process.env.EMBED_MODEL || "text-embedding-nomic-embed-text-v1.5";
const EMBED_ENDPOINT = "http://localhost:1234";
const EMBED_DIM = 768;

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

// Deterministic haystack via seeded PRNG
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
    (r) => `I've been ${pick(["working out","going to the gym","training for a race","doing yoga","practicing martial arts","learning to dance","stretching more","lifting weights","running marathons","swimming laps"])}`,
    (r) => `I need to ${pick(["return this","exchange this","get a refund","file a warranty claim","complain about","report","upgrade","renew","cancel","reactivate"])}${pick([" my subscription","the warranty","the contract","the order","the app","the service","the plan","the membership","the policy","the account"])} at home`,
  ];

  const records = [];
  for (let i = 0; i < count; i++) {
    const text = templates[i % templates.length](i);
    records.push({ id: `h${count}_${i}`, text, theme_id: null, role: "distractor" });
  }
  return records;
}

// Measure cofire: given a list of queries and current record embeddings,
// what fraction of true/near-miss/unrelated pairs appear together in top-k?
function measureCofire(records, queries, true_pairs, near_miss_pairs, unrelated_pairs, k) {
  let trueTotal = 0, trueCofire = 0;
  let nmTotal = 0, nmCofire = 0;
  let unTotal = 0, unCofire = 0;

  for (const q of queries) {
    const ranked = records
      .map((r) => ({ id: r.id, sim: cosine(q.embedding, r.embedding) }))
      .sort((a, b) => b.sim - a.sim)
      .slice(0, k);
    const pids = new Set(ranked.map((r) => r.id));

    for (const tp of true_pairs) {
      trueTotal++;
      if (pids.has(tp.a) && pids.has(tp.b)) trueCofire++;
    }
    for (const nm of near_miss_pairs) {
      nmTotal++;
      if (pids.has(nm.a) && pids.has(nm.b)) nmCofire++;
    }
    for (const up of unrelated_pairs) {
      unTotal++;
      if (pids.has(up.a) && pids.has(up.b)) unCofire++;
    }
  }

  return {
    true_rate: trueTotal > 0 ? trueCofire / trueTotal : 0,
    nm_rate: nmTotal > 0 ? nmCofire / nmTotal : 0,
    un_rate: unTotal > 0 ? unCofire / unTotal : 0,
    true_nm_ratio: nmTotal > 0 && nmCofire > 0
      ? (trueCofire / trueTotal) / (nmCofire / nmTotal)
      : Infinity,
  };
}

// Measure edgeStore bonuses for true/near/unrelated pairs
function measureEdgeBonuses(edgeStore, true_pairs, near_miss_pairs, unrelated_pairs) {
  let trueBonusTotal = 0, trueBonusCount = 0;
  let nmBonusTotal = 0, nmBonusCount = 0;
  let unBonusTotal = 0, unBonusCount = 0;

  for (const tp of true_pairs) {
    const b = edgeStore.bonus(tp.a, tp.b);
    trueBonusTotal += b;
    trueBonusCount++;
  }
  for (const nm of near_miss_pairs) {
    const b = edgeStore.bonus(nm.a, nm.b);
    nmBonusTotal += b;
    nmBonusCount++;
  }
  for (const up of unrelated_pairs) {
    const b = edgeStore.bonus(up.a, up.b);
    unBonusTotal += b;
    unBonusCount++;
  }

  return {
    true_avg_bonus: trueBonusCount > 0 ? trueBonusTotal / trueBonusCount : 0,
    nm_avg_bonus: nmBonusCount > 0 ? nmBonusTotal / nmBonusCount : 0,
    un_avg_bonus: unBonusCount > 0 ? unBonusTotal / unBonusCount : 0,
    true_nm_bonus_ratio: nmBonusCount > 0 ? (trueBonusTotal / trueBonusCount) / (nmBonusTotal / nmBonusCount || 0.0001) : Infinity,
    true_nm_bonus_delta: (trueBonusTotal / trueBonusCount) - (nmBonusTotal / nmBonusCount),
  };
}

async function main() {
  console.log("Loading corpus...");
  const _corpus = JSON.parse(fs.readFileSync(CORPUS_PATH, "utf8"));

  // Make corpus accessible to measure functions
  let corpus = _corpus;

  const nearMissPairSet = new Set(corpus.near_miss_pairs.map((p) => pairKey(p.a, p.b)));
  const truePairSet = new Set(corpus.true_pairs.map((p) => pairKey(p.a, p.b)));
  const unrelatedPairList = corpus.unrelated_pairs;

  const themes = corpus.themes;
  const k = 5;

  // Generate deterministic haystack
  const HAYSTACK_SIZE = 300;
  console.log(`Generating ${HAYSTACK_SIZE} deterministic distractor records (seed=42)...`);
  const haystackRecords = generateHaystackRecords(HAYSTACK_SIZE, 42);

  // Combine records
  const allRecords = [...corpus.records, ...haystackRecords];

  // Embed ALL texts
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

  // Build per-theme query indices
  const themeQueryMap = {};
  for (const theme of themes) {
    const indices = [];
    for (let qi = 0; qi < queries.length; qi++) {
      if (queries[qi].theme_id === theme.id) indices.push(qi);
    }
    themeQueryMap[theme.id] = indices;
  }

  // Pre-reinforcement baseline (cosine-only)
  console.log("\n=== PRE-REINFORCEMENT BASELINE ===");
  const preCofire = measureCofire(records, queries, corpus.true_pairs, corpus.near_miss_pairs, corpus.unrelated_pairs, k);
  console.log(`  True pair rate : ${(preCofire.true_rate * 100).toFixed(2)}%`);
  console.log(`  Near-miss rate : ${(preCofire.nm_rate * 100).toFixed(2)}%`);
  console.log(`  Unrelated rate : ${(preCofire.un_rate * 100).toFixed(2)}%`);
  console.log(`  True/Near ratio: ${preCofire.true_nm_ratio.toFixed(2)}x`);

  // Now run the reinforcement simulation
  console.log("\n=== REINFORCEMENT SIMULATION ===\n");
  console.log("Protocol: cycle through themes, running sessions where user asks");
  console.log("about theme members. Each recall -> reinforceRecall().");
  console.log("After each session batch, measure edgeStore bonuses.\n");

  const edgeStore = new EdgeStore(null);
  const nSessions = 10;
  const convergence = [];

  // Build timeline: for each session, cycle through themes
  // In each theme's session, run ALL its queries (not just one)
  // This simulates a "week" of conversation about that theme
  const timeline = [];
  for (let s = 0; s < nSessions; s++) {
    for (const theme of themes) {
      // Run ALL queries for this theme in this session
      timeline.push({ session: s, theme_id: theme.id, query_indices: themeQueryMap[theme.id] });
    }
  }

  console.log(`Timeline: ${timeline.length} session blocks (${nSessions} sessions x ${themes.length} themes)`);

  for (const step of timeline) {
    // For each query in this theme's session, simulate recall -> reinforce
    for (const qi of step.query_indices) {
      const query = queries[qi];
      const ranked = records
        .map((r) => ({ id: r.id, sim: cosine(query.embedding, r.embedding) }))
        .sort((a, b) => b.sim - a.sim)
        .slice(0, k);
      const topK = ranked.map((r) => r.id);

      // Reinforce (this is where the edgeStore bonus accumulates)
      edgeStore.reinforceRecall(topK, []);
    }

    // Measure after this block
    const edgeMetrics = measureEdgeBonuses(edgeStore, corpus.true_pairs, corpus.near_miss_pairs, corpus.unrelated_pairs);
    const cofireMetrics = measureCofire(records, queries, corpus.true_pairs, corpus.near_miss_pairs, corpus.unrelated_pairs, k);

    convergence.push({
      session: step.session,
      theme_id: step.theme_id,
      global_step: timeline.indexOf(step),
      // EdgeStore metrics
      true_avg_bonus: edgeMetrics.true_avg_bonus,
      nm_avg_bonus: edgeMetrics.nm_avg_bonus,
      un_avg_bonus: edgeMetrics.un_avg_bonus,
      true_nm_bonus_ratio: edgeMetrics.true_nm_bonus_ratio,
      true_nm_bonus_delta: edgeMetrics.true_nm_bonus_delta,
      // Cofire metrics
      cofire_true_rate: cofireMetrics.true_rate,
      cofire_nm_rate: cofireMetrics.nm_rate,
      cofire_un_rate: cofireMetrics.un_rate,
      cofire_true_nm_ratio: cofireMetrics.true_nm_ratio,
    });
  }

  // Final measurement
  const finalEdge = measureEdgeBonuses(edgeStore, corpus.true_pairs, corpus.near_miss_pairs, corpus.unrelated_pairs);
  const finalCofire = measureCofire(records, queries, corpus.true_pairs, corpus.near_miss_pairs, corpus.unrelated_pairs, k);

  // Print convergence
  console.log("\n=== EDGESTORE BONUS CONVERGENCE ===\n");
  console.log(`Session  | True Avg Bonus | Near-Miss Avg Bonus | Unrelated Avg Bonus | True/Near Bonus Ratio`);
  console.log("---------|----------------|---------------------|---------------------|----------------------");

  for (const c of convergence) {
    console.log(
      `${String(c.session).padStart(6)}  | ${c.true_avg_bonus.toFixed(6).padStart(16)} | ${c.nm_avg_bonus.toFixed(6).padStart(19)} | ${c.un_avg_bonus.toFixed(6).padStart(19)} | ${c.true_nm_bonus_ratio.toFixed(2).padStart(22)}x`
    );
  }

  console.log("\n=== COFIRE CONVERGENCE ===\n");
  console.log(`Session  | True Cofire Rate | Near-Miss Cofire | Unrelated Cofire | True/Near Ratio`);
  console.log("---------|------------------|-------------------|-------------------|------------------");

  for (const c of convergence) {
    console.log(
      `${String(c.session).padStart(6)}  | ${(c.cofire_true_rate * 100).toFixed(2).padStart(16)}% | ${(c.cofire_nm_rate * 100).toFixed(2).padStart(19)}% | ${(c.cofire_un_rate * 100).toFixed(2).padStart(19)}% | ${(c.cofire_true_nm_ratio.toFixed(2)).padStart(18)}x`
    );
  }

  // Final verdict
  console.log("\n\n=== FINAL RESULTS ===\n");
  console.log(`[EdgeStore Bonuses - after ${timeline.length} reinforcement steps]`);
  console.log(`  True pair avg bonus :  ${finalEdge.true_avg_bonus.toFixed(6)}`);
  console.log(`  Near-miss avg bonus :  ${finalEdge.nm_avg_bonus.toFixed(6)}`);
  console.log(`  Unrelated avg bonus :  ${finalEdge.un_avg_bonus.toFixed(6)}`);
  console.log(`  True/Near bonus ratio: ${finalEdge.true_nm_bonus_ratio.toFixed(2)}x`);
  console.log(`  True-Near bonus delta: ${finalEdge.true_nm_bonus_delta.toFixed(6)}`);
  console.log();
  console.log(`[Cofire Rates - all queries, post-reinforcement]`);
  console.log(`  True pair rate    : ${(finalCofire.true_rate * 100).toFixed(2)}%`);
  console.log(`  Near-miss rate    : ${(finalCofire.nm_rate * 100).toFixed(2)}%`);
  console.log(`  Unrelated rate    : ${(finalCofire.un_rate * 100).toFixed(2)}%`);
  console.log(`  True/Near cofire ratio: ${finalCofire.true_nm_ratio.toFixed(2)}x`);

  // Per-theme edge metrics
  console.log("\n[Per-theme EdgeStore bonuses after all sessions]");
  const tempEdge = new EdgeStore(null);
  // Reconstruct final state for per-theme
  for (const theme of themes) {
    const themeMembers = corpus.records.filter((r) => r.theme_id === theme.id && r.role === "member");
    let trueBonus = 0, nmBonus = 0, trueCount = 0, nmCount = 0;

    for (const tp of corpus.true_pairs.filter((p) => p.theme_id === theme.id)) {
      const hasMember = (id) => {
        // Check if any theme member is in top-k for ANY query
        for (const qi of themeQueryMap[theme.id]) {
          const query = queries[qi];
          const ranked = records
            .map((r) => ({ id: r.id, sim: cosine(query.embedding, r.embedding) }))
            .sort((a, b) => b.sim - a.sim)
            .slice(0, k);
          const topK = ranked.map((r) => r.id);
          if (topK.includes(id)) return true;
        }
        return false;
      };

      // A true pair gets reinforced when BOTH members appear in top-k
      // of the same query's recall
      let reinforced = false;
      for (const qi of themeQueryMap[theme.id]) {
        const query = queries[qi];
        const ranked = records
          .map((r) => ({ id: r.id, sim: cosine(query.embedding, r.embedding) }))
          .sort((a, b) => b.sim - a.sim)
          .slice(0, k);
        const topK = new Set(ranked.map((r) => r.id));
        if (topK.has(tp.a) && topK.has(tp.b)) reinforced = true;
      }

      trueBonus += reinforced ? tempEdge.bonus(tp.a, tp.b) : 0;
      trueCount++;
    }
    for (const nm of corpus.near_miss_pairs.filter((p) => p.theme_id === theme.id)) {
      let reinforced = false;
      for (const qi of themeQueryMap[theme.id]) {
        const query = queries[qi];
        const ranked = records
          .map((r) => ({ id: r.id, sim: cosine(query.embedding, r.embedding) }))
          .sort((a, b) => b.sim - a.sim)
          .slice(0, k);
        const topK = new Set(ranked.map((r) => r.id));
        if (topK.has(nm.a) && topK.has(nm.b)) reinforced = true;
      }
      nmBonus += reinforced ? tempEdge.bonus(nm.a, nm.b) : 0;
      nmCount++;
    }

    console.log(`  ${theme.name.padEnd(35)} true_bonus=${(finalEdge.true_avg_bonus).toFixed(4)} nm_bonus=${(finalEdge.nm_avg_bonus).toFixed(4)}`);
  }

  console.log("\n=== VERDICT ===");
  if (finalEdge.true_nm_bonus_ratio >= 2.0) {
    console.log("  CLEAR SEPARATION: EdgeStore bonuses strongly favor true pairs.");
    console.log("  RM's bet validated: Hebbian accumulation creates clean separation.");
  } else if (finalEdge.true_nm_bonus_ratio >= 1.3) {
    console.log("  MODERATE SEPARATION: True pairs accumulate more bonus than near-misses.");
    console.log("  The gradient sharpens, but near-misses still get non-trivial reinforcement.");
  } else if (finalEdge.true_nm_bonus_ratio >= 1.0) {
    console.log("  MINIMAL SEPARATION: True and near-miss pairs reinforce at similar rates.");
    console.log("  RM's associative bet is NOT validated — cosine alone can't distinguish them.");
  } else {
    console.log("  REVERSED: Near-misses reinforce MORE than true pairs!");
    console.log("  This is a genuine finding: the associative bet fails under these conditions.");
  }

  const results = {
    version: corpus.version,
    k,
    n_records: allRecords.length,
    n_haystack: HAYSTACK_SIZE,
    n_queries: queries.length,
    embed_model: EMBED_MODEL,
    sessions_per_theme: nSessions,
    pre_reinforcement_cofire: preCofire,
    convergence,
    final_edge: finalEdge,
    final_cofire: finalCofire,
  };

  fs.writeFileSync(RESULTS_PATH, JSON.stringify(results, null, 2));
  console.log(`\nResults saved to ${RESULTS_PATH}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
