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
 * eval/substrate/fire-together-rerun.js — Re-embed the fire-together corpus
 * with nomic-embed-text (nominally the real RM embedder) and add a realistic
 * distractor haystack so top-k is genuinely competitive.
 *
 * Steps:
 *  1. Load the existing corpus (texts, themes, queries, pair labels).
 *  2. Embed ALL texts (records + queries + generated haystack records)
 *     against LM Studio :1234 text-embedding-nomic-embed-text-v1.5.
 *  3. Run the fire-together simulation on the new corpus.
 *  4. Report results — honest, no tuning.
 *
 * Usage:
 *   node eval/substrate/fire-together-rerun.js
 */

"use strict";

const fs = require("fs");
const path = require("path");
const os = require("os");
const { EdgeStore } = require("../../edges.js");
const { cosine } = require("../pipeline.js");

const CORPUS_PATH = path.join(__dirname, "../corpora/fire-together-corpus.json");
const EMBED_ENDPOINT = process.env.EMBED_ENDPOINT || "http://localhost:1234";
const EMBED_MODEL = process.env.EMBED_MODEL || "text-embedding-nomic-embed-text-v1.5";
const EMBED_DIM = 768;

const RESULTS_PATH = path.join(__dirname, "fire-together-rerun-results.json");

// --- Embedding helper ---

async function embedBatch(texts) {
  const body = JSON.stringify({ model: EMBED_MODEL, input: texts });
  const buf = Buffer.from(body);
  const res = await fetch(`${EMBED_ENDPOINT}/v1/embeddings`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "Content-Length": String(buf.length) },
    body: buf,
  });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`Embed failed: ${res.status} ${txt}`);
  }
  const json = JSON.parse(await res.text());
  return json.data.map((d) => d.embedding);
}

async function embedWithBatching(texts, batchSize = 10) {
  const all = [];
  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize);
    console.log(`  embedding batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(texts.length / batchSize)} (${batch.length} texts)`);
    const emb = await embedBatch(batch);
    all.push(...emb);
  }
  return all;
}

// --- Haystack generation ---

function generateHaystackRecords(count = 300) {
  // Realistic distractor texts — personal memories that are thematically
  // distinct but share surface words (clothing, food, places, people)
  // to test whether the separation is genuinely semantic, not lexical.
  const templates = [
    () => `I bought a new ${["blue","red","green","black","white","gray","navy","brown"][Math.floor(Math.random()*8)]} ${["jacket","shirt","pants","dress","sweater","coat","hoodie","scarf","hat","belt"][Math.floor(Math.random()*10)]} from ${["Zara","H&M","Uniqlo","Target","Walmart","Amazon","Nordstrom","Old Navy","Gap","Costco"][Math.floor(Math.random()*10)]} last week`,
    () => `I've been cooking ${["Thai","Italian","Mexican","Japanese","Indian","Korean","French","Chinese","Vietnamese","Lebanese"][Math.floor(Math.random()*10)]} food at home more often lately`,
    () => `My ${["dad","mom","brother","sister","cousin","uncle","aunt","neighbor","friend","colleague"][Math.floor(Math.random()*10)]} lives in ${["Denver","Austin","Portland","Seattle","Chicago","Boston","Miami","Atlanta","Nashville","Minneapolis"][Math.floor(Math.random()*10)]} and we visit once a year`,
    () => `I need to ${["update my resume","schedule a dentist appointment","renew my passport","pay the electric bill","change my oil","fix the leaky faucet","replace the brake pads","clean the gutters","update my insurance","file my taxes"][Math.floor(Math.random()*10)]} before the end of the month`,
    () => `I read a great book about ${["neuroscience","history","cooking","travel","philosophy","psychology","astronomy","genetics","art","politics"][Math.floor(Math.random()*10)]} this past weekend`,
    () => `The ${["parking garage","grocery store","coffee shop","library","gym","hospital","bank","post office","hardware store","pharmacy"][Math.floor(Math.random()*10)]} on ${["Main","Oak","Elm","Maple","Cedar","Pine","Birch","Walnut","Chestnut","Willow"][Math.floor(Math.random()*10)]} Street is always so crowded`,
    () => `I'm trying to cut back on ${["sugar","caffeine","social media","screen time","carbs","dairy","processed food","meat","alcohol","junk food"][Math.floor(Math.random()*10)]}`,
    () => `We had ${["pizza","sushi","tacos","ramen","burgers","pasta","curry","steak","bbq","falafel"]} for dinner last night and it was amazing`,
    () => `My ${["dog","cat","hamster","fish","bird","rabbit","turtle","guinea pig","parrot","ferret"][Math.floor(Math.random()*10)]} ${["loves","hates","enjoys","sleeps in","chases","chews on","perches on","bites","grooms","follows around"]} ${["the couch","the door","the garden","the kitchen","the backyard","the basement","the mailbox","the window","the laundry","the garden hose"]}`,
    () => `I went ${["hiking","swimming","biking","running","kayaking","rock climbing","surfing","fishing","camping","skiing"]} last weekend`,
    () => `I've been listening to a lot of ${["podcasts","lo-fi","jazz","classical","rock","hip-hop","indie","folk","electronic","country"]}, especially ${["The Daily","Lo-Fi Beats","Miles Davis","Beatles","Kendrick","Bon Iver","Daft Punk","Johnny Cash","TED Talks","NPR Music"]}`,
    () => `I ${["love","hate","need","want","miss","enjoy","prefer","avoid","appreciate","dislike"]} ${["morning coffee","late nights","early mornings","weekends","cold weather","hot weather","rain","sunshine","snow","wind"]}`,
    () => `The ${["new","old","local","famous","hidden","chain","independent","corner","downtown","suburban"]}${["restaurant","bar","cafe","bookstore","store","theater","stadium","museum","gallery","studio"]} on ${["5th","1st","Broadway","Grand","Central","Park","Market","Union","Liberty","Independence"]}${["Avenue","Boulevard","Drive","Lane","Way","Court","Place","Road","Street","Plaza"]} is ${["closed","renovating","hiring","expanding","closing down","opening soon","under new management","fully booked","under review","doing well"]}`,
    () => `I should really ${["exercise more","eat better","sleep more","save money","travel more","learn a new skill","call my mom","clean the house","declutter","start meditating"]}`,
    () => `My ${["boss","teacher","professor","manager","coach","supervisor","mentor","advisor","instructor","trainer"]} said I need to ${["improve my communication","work on my patience","be more organized","take more initiative","delegate better","manage my time","be more confident","speak up more","collaborate better","follow through"]}`,
    () => `I went to ${["the beach","the mountains","the lake","the desert","the forest","the canyon","the waterfall","the coastline","the island","the river"]} and the ${["view","weather","water","air"," scenery","nature","wildlife","sunrise","sunset","stars"]} was incredible`,
    () => `I bought a ${["laptop","phone","tablet","monitor","keyboard","headphones","camera","drone","watch","speaker"]} from ${["Apple","Samsung","Dell","Sony","Bose","Canon","DJI","Garmin","JBL","Anker"]}`,
    () => `The ${["traffic","construction","road work","accident","detour","lane closure","bridge repair","utility work","street fair","parade"]} on ${["Highway 101","Interstate 5","Route 66","Main Street","Downtown","The Expressway","The Freeway","The Parkway","The Bypass","The Corridor"]} made me ${["late","take a detour","wait an hour","find an alternate route","lose my patience","call in late","get stuck","turn around","replan","sit in traffic"]}`,
    () => `I've been ${["working out","going to the gym","training for a race","doing yoga","practicing martial arts","learning to dance","stretching more","lifting weights","running marathons","swimming laps"]}`,
    () => `I need to ${["return this","exchange this","get a refund","file a warranty claim","complain about","report","upgrade","renew","cancel","reactivate"]}${[" my subscription","the warranty","the contract","the order","the app","the service","the plan","the membership","the policy","the account"]}`,
  ];

  const records = [];
  for (let i = 0; i < count; i++) {
    const fn = templates[i % templates.length];
    // Make each text slightly different by adding random noise words
    const base = fn();
    const noise = ["actually","really","just","sort of","kind of","pretty much","totally","basically","honestly","frankly"];
    const noiseWord = noise[Math.floor(Math.random() * noise.length)];
    const noisePos = Math.random() < 0.5 ? 0 : base.length;
    const text = noisePos === 0
      ? `${noiseWord.charAt(0).toUpperCase() + noiseWord.slice(1)} ${base}`
      : `${base} ${noiseWord}.`;
    records.push({
      id: `h_${i}`,
      text,
      theme_id: null, // haystack records have no theme
      role: "distractor",
    });
  }
  return records;
}

// --- Run the simulation (identical to fire-together-sim.js) ---

function runSimulation(records, queries, nearMissPairSet, truePairSet, k = 5, corpusInfo = {}, opts = {}) {
  const corpus = corpusInfo.corpus;
  const themes = corpusInfo.themes;
  const haystackRecords = corpusInfo.haystackRecords;
  const nQueries = queries.length;
  const traces = [];
  const cofireCounts = {};
  const edgeStore = new EdgeStore(null);
  let h1Hits = 0;
  let falseCoactivations = 0;
  let offDiagonalLeakageHits = 0;

  for (let qIdx = 0; qIdx < queries.length; qIdx++) {
    const q = queries[qIdx];
    const ranked = records
      .map((r) => ({
        id: r.id,
        text: r.text,
        sim: cosine(q.embedding, r.embedding),
        role: r.role,
        theme_id: r.theme_id,
      }))
      .sort((a, b) => b.sim - a.sim)
      .slice(0, k);

    const primaryIds = ranked.map((r) => r.id);

    let h1Hit = false;
    if (q.target_id) {
      h1Hit = primaryIds.includes(q.target_id);
    } else if (q.theme_id) {
      const themeMembers = records
        .filter((r) => r.theme_id === q.theme_id && r.role === "member")
        .map((r) => r.id);
      h1Hit = primaryIds.some((id) => themeMembers.includes(id));
    }
    if (h1Hit) h1Hits++;

    const coactivePairs = [];
    const edgeDeltas = [];

    for (let i = 0; i < primaryIds.length; i++) {
      for (let j = i + 1; j < primaryIds.length; j++) {
        const idA = primaryIds[i];
        const idB = primaryIds[j];
        const pKey = (String(idA) < String(idB) ? idA + "\t" + idB : idB + "\t" + idA);
        cofireCounts[pKey] = (cofireCounts[pKey] || 0) + 1;
        coactivePairs.push([idA, idB]);

        if (nearMissPairSet.has(pKey)) falseCoactivations++;
        if (!truePairSet.has(pKey) && !nearMissPairSet.has(pKey)) offDiagonalLeakageHits++;

        const bonusBefore = edgeStore.bonus(idA, idB);
        edgeDeltas.push({ pair: [idA, idB], before: bonusBefore });
      }
    }

    edgeStore.reinforceRecall(primaryIds, []);

    for (const d of edgeDeltas) {
      d.after = edgeStore.bonus(d.pair[0], d.pair[1]);
      d.delta = d.after - d.before;
    }

    traces.push({
      query_idx: qIdx,
      query: q.query,
      kind: q.kind,
      theme_id: q.theme_id,
      target_id: q.target_id,
      h1_hit: h1Hit,
      primary_top_k: primaryIds,
      coactive_pairs: coactivePairs,
      edge_deltas: edgeDeltas,
    });
  }

  // --- Compile Metrics ---

  const h1Rate = h1Hits / nQueries;
  const h1Paraphrases = queries.filter((q) => q.kind === "member-paraphrase");
  const h1ParaphraseHits = traces.filter((t) => t.kind === "member-paraphrase" && t.h1_hit).length;
  const h1ParaphraseRate = h1ParaphraseHits / (h1Paraphrases.length || 1);
  const h1Status = h1Rate >= 0.80 ? "PASS" : (h1Rate >= 0.50 ? "HUNGRY" : "STARVE");

  const truePairRates = corpus.true_pairs.map((p) => {
    const key = (p.a < p.b ? p.a + "\t" + p.b : p.b + "\t" + p.a);
    const count = cofireCounts[key] || 0;
    return { a: p.a, b: p.b, pair: `${p.a} ↔ ${p.b}`, theme_id: p.theme_id, count, rate: count / nQueries };
  });
  const trueMeanRate = truePairRates.reduce((s, r) => s + r.rate, 0) / (truePairRates.length || 1);
  const themesOver30 = {};
  for (const t of themes) {
    const tp = truePairRates.filter((r) => r.theme_id === t.id);
    const maxR = tp.length ? Math.max(...tp.map((r) => r.rate)) : 0;
    themesOver30[t.id] = { name: t.name, max_rate: maxR, over30: maxR >= 0.30 };
  }
  const h2Pass = trueMeanRate >= 0.25 && Object.values(themesOver30).filter((t) => t.over30).length >= 3;
  const h2Status = h2Pass ? "PASS" : (trueMeanRate >= 0.10 ? "HUNGRY" : "STARVE");

  const nearMissRates = corpus.near_miss_pairs.map((p) => {
    const key = (p.a < p.b ? p.a + "\t" + p.b : p.b + "\t" + p.a);
    const count = cofireCounts[key] || 0;
    return { a: p.a, b: p.b, pair: `${p.a} ↔ ${p.b}`, theme_id: p.theme_id, count, rate: count / nQueries };
  });
  const nearMissMeanRate = nearMissRates.reduce((s, r) => s + r.rate, 0) / (nearMissRates.length || 1);
  const nearMissMaxRate = nearMissRates.length ? Math.max(...nearMissRates.map((r) => r.rate)) : 0;
  const h3Pass = nearMissMeanRate <= 0.10 && nearMissMaxRate < 0.25;
  const h3Status = h3Pass ? "PASS" : (nearMissMaxRate < 0.25 ? "CONCERN" : "FAIL");

  const unrelatedRates = corpus.unrelated_pairs.map((p) => {
    const key = (p.a < p.b ? p.a + "\t" + p.b : p.b + "\t" + p.a);
    const count = cofireCounts[key] || 0;
    return { a: p.a, b: p.b, pair: `${p.a} ↔ ${p.b}`, count, rate: count / nQueries };
  });
  const leakageMeanRate = unrelatedRates.reduce((s, r) => s + r.rate, 0) / (unrelatedRates.length || 1);
  const leakageMaxRate = unrelatedRates.length ? Math.max(...unrelatedRates.map((r) => r.rate)) : 0;
  const h4Pass = leakageMeanRate <= 0.05;
  const h4Status = h4Pass ? "PASS" : "FAIL";

  const trueInBand = corpus.true_pairs.filter((p) => edgeStore.bonus(p.a, p.b) >= 0.05);
  const nmInBand = corpus.near_miss_pairs.filter((p) => edgeStore.bonus(p.a, p.b) >= 0.05);
  const unInBand = corpus.unrelated_pairs.filter((p) => edgeStore.bonus(p.a, p.b) >= 0.05);
  const trueBandFrac = trueInBand.length / (corpus.true_pairs.length || 1);
  const nmBandFrac = nmInBand.length / (corpus.near_miss_pairs.length || 1);
  const unBandFrac = unInBand.length / (corpus.unrelated_pairs.length || 1);
  const h5Pass = trueBandFrac > unBandFrac;
  const h5Status = h5Pass ? "PASS" : "FAIL";

  return {
    version: corpus.version,
    k,
    n_records: records.length,
    n_haystack: haystackRecords.length,
    n_queries: nQueries,
    embed_model: EMBED_MODEL,
    hypotheses: {
      H1_retrieval: {
        description: "Paraphrase puts its fact in top-5",
        overall_rate: h1Rate,
        paraphrase_rate: h1ParaphraseRate,
        status: h1Status,
      },
      H2_cofire: {
        description: "Mean pairwise cofire_rate on true theme pairs",
        mean_rate: trueMeanRate,
        themes_over_30: `${Object.values(themesOver30).filter((t) => t.over30).length}/5`,
        theme_max_rates: themesOver30,
        true_pairs: truePairRates,
        threshold_pass: "mean >= 0.25 and >=3 themes >= 0.30",
        status: h2Status,
      },
      H3_near_miss_cofire: {
        description: "Mean cofire of labeled near-miss pairs",
        mean_rate: nearMissMeanRate,
        max_rate: nearMissMaxRate,
        threshold_pass: "mean <= 0.10 and no pair >= 0.25",
        status: h3Status,
      },
      H4_leakage: {
        description: "Mean cofire of unrelated (no-theme, no-near-miss) pairs",
        mean_rate: leakageMeanRate,
        max_rate: leakageMaxRate,
        threshold_pass: "<= 0.05",
        status: h4Status,
      },
      H5_edge_delta: {
        description: "True pairs vs unrelated reaching bonus >= 0.05 band",
        true_pairs_in_band: `${trueInBand.length}/${corpus.true_pairs.length} (${(trueBandFrac * 100).toFixed(1)}%)`,
        near_miss_in_band: `${nmInBand.length}/${corpus.near_miss_pairs.length} (${(nmBandFrac * 100).toFixed(1)}%)`,
        unrelated_in_band: `${unInBand.length}/${corpus.unrelated_pairs.length} (${(unBandFrac * 100).toFixed(1)}%)`,
        status: h5Status,
      },
    },
    cofire_counts: cofireCounts,
    traces: opts.includeTraces ? traces : undefined,
  };
}

function printReport(results) {
  const h = results.hypotheses;
  console.log("\n================================================================================");
  console.log("     FIRE-TOGETHER SIMULATION — nomic-embed-text-v1.5 + realistic haystack      ");
  console.log("================================================================================");
  console.log(`Embedder       : ${results.embed_model}`);
  console.log(`Records        : ${results.n_records} total (${results.n_haystack} haystack distractors + 25 original)`);
  console.log(`Queries        : ${results.n_queries} natural phrasings`);
  console.log(`Top-k          : ${results.k}`);
  console.log("--------------------------------------------------------------------------------");
  console.log("Hypothesis Results:\n");

  console.log(`[H1 Retrieval — paraphrase puts fact in top-5]`);
  console.log(`  Paraphrase hit rate: ${(h.H1_retrieval.paraphrase_rate * 100).toFixed(1)}%  Verdict: ${h.H1_retrieval.status}`);

  console.log(`\n[H2 Cofire Headline — true theme pairs co-fire]`);
  console.log(`  Mean cofire rate : ${(h.H2_cofire.mean_rate * 100).toFixed(2)}%`);
  console.log(`  Themes >= 0.30   : ${h.H2_cofire.themes_over_30}`);
  console.log(`  Verdict          : ${h.H2_cofire.status}`);

  console.log(`\n[H3 Near-Miss Cofire — lookalikes that share surface words]`);
  console.log(`  Mean cofire rate : ${(h.H3_near_miss_cofire.mean_rate * 100).toFixed(2)}%`);
  console.log(`  Max pair rate    : ${(h.H3_near_miss_cofire.max_rate * 100).toFixed(2)}%`);
  console.log(`  Verdict          : ${h.H3_near_miss_cofire.status}`);

  console.log(`\n[H4 Unrelated Leakage — no-theme pairs]`);
  console.log(`  Mean cofire rate : ${(h.H4_leakage.mean_rate * 100).toFixed(2)}%`);
  console.log(`  Verdict          : ${h.H4_leakage.status}`);

  console.log(`\n[H5 Edge Delta — Hebbian reinforcement after simulated week]`);
  console.log(`  True pairs in band: ${h.H5_edge_delta.true_pairs_in_band}`);
  console.log(`  Near-miss in band : ${h.H5_edge_delta.near_miss_in_band}`);
  console.log(`  Unrelated in band : ${h.H5_edge_delta.unrelated_in_band}`);
  console.log(`  Verdict           : ${h.H5_edge_delta.status}`);

  console.log("\n--------------------------------------------------------------------------------");
  console.log("Key comparison — H2 (true pairs) vs H3 (near-misses):\n");
  console.log(`  True pair mean rate : ${(h.H2_cofire.mean_rate * 100).toFixed(2)}%`);
  console.log(`  Near-miss mean rate : ${(h.H3_near_miss_cofire.mean_rate * 100).toFixed(2)}%`);
  console.log(`  Delta               : ${((h.H2_cofire.mean_rate - h.H3_near_miss_cofire.mean_rate) * 100).toFixed(2)}%`);
  console.log(`  Ratio (true/near)   : ${(h.H2_cofire.mean_rate / (h.H3_near_miss_cofire.mean_rate || 0.001)).toFixed(2)}x`);

  console.log("\nNear-miss pair details:");
  for (const p of h.H3_near_miss_cofire.themes_over_30 !== undefined
    ? corpus.near_miss_pairs.map((p, i) => ({
        ...p, rate: h.H3_near_miss_cofire.mean_rate, _idx: i
      }))
    : corpus.near_miss_pairs) {
    // We don't have per-pair near miss rates in this printout, add below
  }

  console.log("\n================================================================================");
}

// --- Main ---

async function main() {
  console.log("Loading corpus...");
  const corpus = JSON.parse(fs.readFileSync(CORPUS_PATH, "utf8"));
  const themes = corpus.themes;
  const nearMissPairSet = new Set(corpus.near_miss_pairs.map((p) => {
    const a = String(p.a), b = String(p.b);
    return a < b ? a + "\t" + b : b + "\t" + a;
  }));
  const truePairSet = new Set(corpus.true_pairs.map((p) => {
    const a = String(p.a), b = String(p.b);
    return a < b ? a + "\t" + b : b + "\t" + a;
  }));

  console.log(`Original corpus: ${corpus.records.length} records, ${corpus.queries.length} queries`);
  console.log(`Embedding model: ${EMBED_MODEL} @ ${EMBED_ENDPOINT}`);

  // Generate haystack
  const haystackCount = 300;
  console.log(`\nGenerating ${haystackCount} realistic distractor records...`);
  const haystackRecords = generateHaystackRecords(haystackCount);

  // Combine records
  const allRecords = [...corpus.records, ...haystackRecords];

  // Embed ALL texts (records + queries)
  const allTexts = [...allRecords.map((r) => r.text), ...corpus.queries.map((q) => q.query)];
  const totalTexts = allTexts.length;
  console.log(`Embedding ${totalTexts} texts (${allRecords.length} records + ${corpus.queries.length} queries)...`);

  const allEmbeddings = await embedWithBatching(allTexts, 10);

  // Assign embeddings back
  let recOffset = 0;
  const updatedRecords = allRecords.map((r) => ({
    ...r,
    embedding: allEmbeddings[recOffset++],
  }));

  const updatedQueries = corpus.queries.map((q) => ({
    ...q,
    embedding: allEmbeddings[recOffset++],
  }));

  console.log("Embedding complete. Running simulation...");

  const results = runSimulation(updatedRecords, updatedQueries, nearMissPairSet, truePairSet, 5, { corpus, themes, haystackRecords });

  // Save results
  fs.writeFileSync(RESULTS_PATH, JSON.stringify(results, null, 2));
  console.log(`\nResults saved to ${RESULTS_PATH}`);

  printReport(results);

  // Per-theme details
  console.log("\n--- Per-theme true pair max rates ---");
  for (const t of themes) {
    const tp = results.hypotheses.H2_cofire.true_pairs.filter((p) => p.theme_id === t.id);
    const maxRate = tp.length ? Math.max(...tp.map((p) => p.rate)) : 0;
    console.log(`  ${t.name.padEnd(30)} max: ${(maxRate * 100).toFixed(1)}%`);
  }

  // Near-miss pair details
  console.log("\n--- Per-near-miss pair rates ---");
  const nmKey = (a, b) => { const x = String(a), y = String(b); return x < y ? x + "\t" + y : y + "\t" + x; };
  for (const p of corpus.near_miss_pairs) {
    const key = nmKey(p.a, p.b);
    const count = results.cofire_counts[key] || 0;
    const rate = count / results.n_queries;
    console.log(`  ${p.a} <-> ${p.b}: ${(rate * 100).toFixed(1)}% (${count}/${results.n_queries})`);
  }

  // Best/worst case — what does the top-5 look like for a member-paraphrase query?
  console.log("\n--- Sample query traces (member-paraphrase) ---");
  const traces = results.traces.filter((t) => t.kind === "member-paraphrase").slice(0, 5);
  for (const t of traces) {
    console.log(`  Query: "${t.query}"`);
    // Need to recompute sims for display
    console.log("  (top-k IDs — see results JSON for full data)");
    console.log("");
  }

  console.log("Done.");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
