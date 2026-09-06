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
 * eval/substrate/fire-together-sim.js — 4.0b Hermes Fire-Together Simulation (RFC 0011 §7.2)
 *
 * Runs deterministic measurement over the committed fire-together corpus
 * (eval/corpora/fire-together-corpus.json).
 *
 * Evaluates the 5 pre-declared hypotheses:
 *   H1: Retrieval (paraphrase puts its fact in top-5, pass >= 0.80)
 *   H2: Cofire headline (mean pairwise cofire on true theme pairs, pass >= 0.25 & >=3 themes >= 0.30)
 *   H3: Near-miss cofire (mean cofire of labeled near-miss pairs, pass <= 0.10 & no pair >= 0.25)
 *   H4: Leakage (mean cofire of unrelated pairs, pass <= 0.05)
 *   H5: Edge delta (true pairs reaching bonus >= 0.05 vs unrelated pairs)
 *
 * Usage:
 *   node eval/substrate/fire-together-sim.js
 *   node eval/substrate/fire-together-sim.js --json
 *   node eval/substrate/fire-together-sim.js --verbose
 */

"use strict";

const fs = require("fs");
const path = require("path");
const os = require("os");
const { EdgeStore } = require("../../edges.js");
const { JsonlStore } = require("../../store.js");
const { createMemory, cosine } = require("../pipeline.js");
const { computeMetric, explainMetric } = require("../metrics.js");

const CORPUS_PATH = path.join(__dirname, "../corpora/fire-together-corpus.json");
const DEFAULT_RESULTS_PATH = path.join(__dirname, "fire-together-results.json");

function pairKey(a, b) {
  const x = String(a), y = String(b);
  return x < y ? x + "\t" + y : y + "\t" + x;
}

function runSimulation(corpus, opts = {}) {
  const k = opts.k || 5;
  const records = corpus.records;
  const queries = corpus.queries;
  const nQueries = queries.length;

  const recordsById = new Map(records.map((r) => [String(r.id), r]));
  const nearMissPairSet = new Set(corpus.near_miss_pairs.map((p) => pairKey(p.a, p.b)));
  const truePairSet = new Set(corpus.true_pairs.map((p) => pairKey(p.a, p.b)));

  // In-memory EdgeStore to track Hebbian reinforcement deltas
  const edgeStore = new EdgeStore(null);

  // Per-query traces and co-activation matrix
  const traces = [];
  const cofireCounts = {};
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

    // H1 check
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

    // Co-activation pairs & edge deltas
    const coactivePairs = [];
    const edgeDeltas = [];

    for (let i = 0; i < primaryIds.length; i++) {
      for (let j = i + 1; j < primaryIds.length; j++) {
        const idA = primaryIds[i];
        const idB = primaryIds[j];
        const pKey = pairKey(idA, idB);
        cofireCounts[pKey] = (cofireCounts[pKey] || 0) + 1;
        coactivePairs.push([idA, idB]);

        if (nearMissPairSet.has(pKey)) {
          falseCoactivations++;
        }
        if (!truePairSet.has(pKey) && !nearMissPairSet.has(pKey)) {
          offDiagonalLeakageHits++;
        }

        const bonusBefore = edgeStore.bonus(idA, idB);
        edgeDeltas.push({ pair: [idA, idB], before: bonusBefore });
      }
    }

    // Reinforce recall in EdgeStore (simulating natural conversational week)
    edgeStore.reinforceRecall(primaryIds, []);

    // Record after deltas
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

  // --- Compile Pre-Declared Metrics (H1–H5) ---

  // H1: Retrieval
  const h1Rate = h1Hits / nQueries;
  const h1Paraphrases = queries.filter((q) => q.kind === "member-paraphrase");
  const h1ParaphraseHits = traces
    .filter((t) => t.kind === "member-paraphrase" && t.h1_hit)
    .length;
  const h1ParaphraseRate = h1ParaphraseHits / (h1Paraphrases.length || 1);
  const h1Status = h1Rate >= 0.80 ? "PASS" : (h1Rate >= 0.50 ? "HUNGRY" : "STARVE");

  // H2: Cofire (true theme pairs)
  const truePairRates = corpus.true_pairs.map((p) => {
    const key = pairKey(p.a, p.b);
    const count = cofireCounts[key] || 0;
    return {
      a: p.a,
      b: p.b,
      pair: `${p.a} ↔ ${p.b}`,
      theme_id: p.theme_id,
      count,
      rate: count / nQueries,
    };
  });
  const trueMeanRate = truePairRates.reduce((s, r) => s + r.rate, 0) / (truePairRates.length || 1);

  // Per-theme check: >= 3/5 themes have a pair >= 0.30
  const themeMaxRates = {};
  for (const t of corpus.themes) {
    const themePairs = truePairRates.filter((r) => r.theme_id === t.id);
    const maxR = themePairs.length ? Math.max(...themePairs.map((r) => r.rate)) : 0;
    themeMaxRates[t.id] = { name: t.name, max_rate: maxR };
  }
  const themesOver30 = Object.values(themeMaxRates).filter((t) => t.max_rate >= 0.30).length;
  const h2Pass = trueMeanRate >= 0.25 && themesOver30 >= 3;
  const h2Hungry = trueMeanRate >= 0.10;
  const h2Status = h2Pass ? "PASS" : (h2Hungry ? "HUNGRY" : "STARVE");

  // H3: Near-miss cofire
  const nearMissRates = corpus.near_miss_pairs.map((p) => {
    const key = pairKey(p.a, p.b);
    const count = cofireCounts[key] || 0;
    return {
      a: p.a,
      b: p.b,
      pair: `${p.a} ↔ ${p.b}`,
      theme_id: p.theme_id,
      count,
      rate: count / nQueries,
    };
  });
  const nearMissMeanRate = nearMissRates.reduce((s, r) => s + r.rate, 0) / (nearMissRates.length || 1);
  const nearMissMaxRate = nearMissRates.length ? Math.max(...nearMissRates.map((r) => r.rate)) : 0;
  const h3Pass = nearMissMeanRate <= 0.10 && nearMissMaxRate < 0.25;
  const h3Status = h3Pass ? "PASS" : (nearMissMaxRate < 0.25 ? "CONCERN" : "FAIL");

  // H4: Leakage (unrelated pairs)
  const unrelatedRates = corpus.unrelated_pairs.map((p) => {
    const key = pairKey(p.a, p.b);
    const count = cofireCounts[key] || 0;
    return {
      a: p.a,
      b: p.b,
      pair: `${p.a} ↔ ${p.b}`,
      count,
      rate: count / nQueries,
    };
  });
  const leakageMeanRate = unrelatedRates.reduce((s, r) => s + r.rate, 0) / (unrelatedRates.length || 1);
  const leakageMaxRate = unrelatedRates.length ? Math.max(...unrelatedRates.map((r) => r.rate)) : 0;
  const h4Pass = leakageMeanRate <= 0.05;
  const h4Status = h4Pass ? "PASS" : "FAIL";

  // H5: Edge delta after scripted week (bonus >= 0.05)
  const trueInBand = corpus.true_pairs.filter((p) => edgeStore.bonus(p.a, p.b) >= 0.05);
  const nmInBand = corpus.near_miss_pairs.filter((p) => edgeStore.bonus(p.a, p.b) >= 0.05);
  const unInBand = corpus.unrelated_pairs.filter((p) => edgeStore.bonus(p.a, p.b) >= 0.05);

  const trueBandFraction = trueInBand.length / (corpus.true_pairs.length || 1);
  const nmBandFraction = nmInBand.length / (corpus.near_miss_pairs.length || 1);
  const unBandFraction = unInBand.length / (corpus.unrelated_pairs.length || 1);

  const h5Pass = trueBandFraction > unBandFraction;
  const h5Status = h5Pass ? "PASS" : "FAIL";

  // Integration with eval/metrics.js standard registers
  const registeredCofire = computeMetric("cofire_rate", {
    cofire: { counts: cofireCounts, n_queries: nQueries },
    true_pairs: corpus.true_pairs,
  });
  const registeredNearMiss = computeMetric("near_miss_cofire", {
    cofire: { counts: cofireCounts, n_queries: nQueries },
    near_miss_pairs: corpus.near_miss_pairs,
  });
  const registeredLeakage = computeMetric("leakage_cofire", {
    cofire: { counts: cofireCounts, n_queries: nQueries },
    unrelated_pairs: corpus.unrelated_pairs,
  });

  return {
    version: corpus.version,
    k,
    n_records: records.length,
    n_queries: nQueries,
    hypotheses: {
      H1_retrieval: {
        description: "Paraphrase puts its fact in top-5",
        overall_rate: h1Rate,
        paraphrase_rate: h1ParaphraseRate,
        threshold_pass: ">= 0.80",
        threshold_starve: "< 0.50",
        status: h1Status,
      },
      H2_cofire: {
        description: "Mean pairwise cofire_rate on true theme pairs",
        mean_rate: trueMeanRate,
        registered_metric: registeredCofire,
        themes_over_30: `${themesOver30}/5`,
        theme_max_rates: themeMaxRates,
        true_pairs: truePairRates,
        threshold_pass: "mean >= 0.25 and >=3 themes >= 0.30",
        threshold_starve: "< 0.10",
        status: h2Status,
        band_note: "0.10–0.25 is usable but hungry",
      },
      H3_near_miss_cofire: {
        description: "Mean cofire of labeled near-miss pairs",
        mean_rate: nearMissMeanRate,
        max_rate: nearMissMaxRate,
        registered_metric: registeredNearMiss,
        threshold_pass: "mean <= 0.10 and no pair >= 0.25",
        status: h3Status,
      },
      H4_leakage: {
        description: "Mean cofire of unrelated (no-theme, no-near-miss) pairs",
        mean_rate: leakageMeanRate,
        max_rate: leakageMaxRate,
        registered_metric: registeredLeakage,
        threshold_pass: "<= 0.05",
        status: h4Status,
      },
      H5_edge_delta: {
        description: "True pairs vs unrelated reaching bonus >= 0.05 band",
        true_pairs_in_band: `${trueInBand.length}/${corpus.true_pairs.length} (${(trueBandFraction * 100).toFixed(1)}%)`,
        near_miss_in_band: `${nmInBand.length}/${corpus.near_miss_pairs.length} (${(nmBandFraction * 100).toFixed(1)}%)`,
        unrelated_in_band: `${unInBand.length}/${corpus.unrelated_pairs.length} (${(unBandFraction * 100).toFixed(1)}%)`,
        status: h5Status,
      },
    },
    cofire_counts: cofireCounts,
    traces: opts.includeTraces ? traces : undefined,
  };
}

function printReport(results, verbose = false) {
  const h = results.hypotheses;
  console.log("\n================================================================================");
  console.log("             RFC 0011 §7.2 — 4.0b Hermes Fire-Together Simulation               ");
  console.log("================================================================================");
  console.log(`Corpus Version : ${results.version}`);
  console.log(`Records        : ${results.n_records} (15 on-theme, 5 near-miss, 5 distractors)`);
  console.log(`Queries        : ${results.n_queries} natural phrasings`);
  console.log(`Top-k          : ${results.k}`);
  console.log("--------------------------------------------------------------------------------");
  console.log("Pre-Declared Hypotheses & Measurement Results:\n");

  // H1
  console.log(`[H1 Retrieval]`);
  console.log(`  Description   : ${h.H1_retrieval.description}`);
  console.log(`  Paraphrases   : ${(h.H1_retrieval.paraphrase_rate * 100).toFixed(2)}%`);
  console.log(`  Overall       : ${(h.H1_retrieval.overall_rate * 100).toFixed(2)}%`);
  console.log(`  Pass Band     : ${h.H1_retrieval.threshold_pass}`);
  console.log(`  Verdict       : [ ${h.H1_retrieval.status} ]\n`);

  // H2
  console.log(`[H2 Cofire Headline]`);
  console.log(`  Description   : ${h.H2_cofire.description}`);
  console.log(`  Mean Cofire   : ${h.H2_cofire.mean_rate.toFixed(4)} (registered metric: ${h.H2_cofire.registered_metric.toFixed(4)})`);
  console.log(`  Themes >= 0.30: ${h.H2_cofire.themes_over_30}`);
  console.log(`  Pass Band     : ${h.H2_cofire.threshold_pass}`);
  console.log(`  Verdict       : [ ${h.H2_cofire.status} ] (${h.H2_cofire.band_note})\n`);

  // H3
  console.log(`[H3 Near-Miss Cofire]`);
  console.log(`  Description   : ${h.H3_near_miss_cofire.description}`);
  console.log(`  Mean Cofire   : ${h.H3_near_miss_cofire.mean_rate.toFixed(4)} (registered metric: ${h.H3_near_miss_cofire.registered_metric.toFixed(4)})`);
  console.log(`  Max Pair Rate : ${h.H3_near_miss_cofire.max_rate.toFixed(4)} (threshold: no pair >= 0.25)`);
  console.log(`  Verdict       : [ ${h.H3_near_miss_cofire.status} ]\n`);

  // H4
  console.log(`[H4 Unrelated Leakage]`);
  console.log(`  Description   : ${h.H4_leakage.description}`);
  console.log(`  Mean Leakage  : ${h.H4_leakage.mean_rate.toFixed(4)} (registered metric: ${h.H4_leakage.registered_metric.toFixed(4)})`);
  console.log(`  Pass Band     : ${h.H4_leakage.threshold_pass}`);
  console.log(`  Verdict       : [ ${h.H4_leakage.status} ]\n`);

  // H5
  console.log(`[H5 Edge Delta]`);
  console.log(`  Description   : ${h.H5_edge_delta.description}`);
  console.log(`  True Pairs    : ${h.H5_edge_delta.true_pairs_in_band}`);
  console.log(`  Near-Miss     : ${h.H5_edge_delta.near_miss_in_band}`);
  console.log(`  Unrelated     : ${h.H5_edge_delta.unrelated_in_band}`);
  console.log(`  Verdict       : [ ${h.H5_edge_delta.status} ] (true pairs reach band more than unrelated)\n`);

  if (verbose) {
    console.log("--------------------------------------------------------------------------------");
    console.log("True Pair Details:");
    for (const p of h.H2_cofire.true_pairs) {
      console.log(`  ${p.pair.padEnd(20)} theme=${p.theme_id.padEnd(16)} count=${String(p.count).padStart(3)} rate=${p.rate.toFixed(4)}`);
    }
    console.log("\nTheme Max Rates:");
    for (const [tid, info] of Object.entries(h.H2_cofire.theme_max_rates)) {
      console.log(`  ${tid.padEnd(16)} (${info.name}): max_rate=${info.max_rate.toFixed(4)}`);
    }
  }

  console.log("================================================================================");
  console.log("RFC 0011 §7.2 Rule: The sim REPORTS, never auto-writes constants.");
  console.log("================================================================================\n");
}

function main() {
  const args = process.argv.slice(2);
  const jsonMode = args.includes("--json");
  const verbose = args.includes("--verbose");

  if (!fs.existsSync(CORPUS_PATH)) {
    console.error(`Error: Corpus not found at ${CORPUS_PATH}. Run generator first.`);
    process.exit(1);
  }

  const corpus = JSON.parse(fs.readFileSync(CORPUS_PATH, "utf8"));
  const results = runSimulation(corpus, { includeTraces: false });

  if (jsonMode) {
    console.log(JSON.stringify(results, null, 2));
  } else {
    printReport(results, verbose);
  }

  fs.writeFileSync(DEFAULT_RESULTS_PATH, JSON.stringify(results, null, 2));
}

if (require.main === module) {
  main();
}

module.exports = { runSimulation, CORPUS_PATH };
