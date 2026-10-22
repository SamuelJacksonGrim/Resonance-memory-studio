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
 * eval/substrate/fire-together-decay.js — Equilibrium reinforcement experiment.
 *
 * Tests RM's core bet: does Hebbian reinforcement create meaningful separation
 * between true thematic pairs and near-miss pairs under wall-clock decay?
 *
 * The stored weight in EdgeStore grows unboundedly because within-session
 * reinforcement events are milliseconds apart (negligible decay). Between
 * sessions, there's a 12-24h gap. The stored weight accumulates because
 * each session adds alpha, and decay between sessions is limited.
 *
 * bonus() = tanh(effectiveHebbian) * maxBonus (0.3).
 * effectiveHebbian = stored_weight * 2^(-dt/H).
 *
 * The KEY question: at the time of a recall query (right after reinforcement
 * of the previous query), what is the bonus for true pairs vs near-misses?
 *
 * Measurement protocol:
 *   After each session, measure bonus() at time = "now" (immediately after
 *   last reinforcement). This simulates the state at the start of the next
 *   session, when the user would query again.
 *
 *   Also measure at 1h, 6h, 24h, 7d gaps to show decay.
 */

"use strict";

const fs = require("fs");
const path = require("path");
const { EdgeStore } = require("../../edges.js");
const { cosine } = require("../pipeline.js");
const { effectiveHebbian } = require("../../edges.js");

const CORPUS_PATH = path.join(__dirname, "../corpora/fire-together-corpus.json");
const RESULTS_PATH = path.join(__dirname, "fire-together-decay-results.json");
const EMBED_MODEL = process.env.EMBED_MODEL || "text-embedding-nomic-embed-text-v1.5";
const EMBED_ENDPOINT = "http://localhost:1234";

const DAY_SECS = 86400;
const HOUR_SECS = 3600;
const HALF_LIFE_SECS = 7 * DAY_SECS;

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

// Measure effectiveHebbian (which is what bonus() uses before tanh)
// at a given time offset from "now". The stored weight was last updated
// at "now" (this session). At dtSecs offset, effectiveHebbian = stored * 2^(-dtSecs/H).
function measureEffectiveAtOffset(edgeStore, offsetsSecs, truePairs, nearMissPairs, unrelatedPairs) {
  const result = {};

  for (const dtSecs of offsetsSecs) {
    let trueSum = 0, nmSum = 0, unSum = 0;
    let trueCount = 0, nmCount = 0, unCount = 0;

    for (const tp of truePairs) {
      const w = edgeStore.weight(tp.a, tp.b);
      const boosted = w * Math.pow(2, -dtSecs / HALF_LIFE_SECS);
      trueSum += boosted;
      trueCount++;
    }
    for (const nm of nearMissPairs) {
      const w = edgeStore.weight(nm.a, nm.b);
      const boosted = w * Math.pow(2, -dtSecs / HALF_LIFE_SECS);
      nmSum += boosted;
      nmCount++;
    }
    for (const up of unrelatedPairs) {
      const w = edgeStore.weight(up.a, up.b);
      const boosted = w * Math.pow(2, -dtSecs / HALF_LIFE_SECS);
      unSum += boosted;
      unCount++;
    }

    const key = dtSecs === 0 ? "now" : dtSecs <= HOUR_SECS ? `+${(dtSecs / 3600).toFixed(1)}h` : `+${(dtSecs / DAY_SECS).toFixed(1)}d`;
    result[key] = {
      true_avg: trueSum / trueCount,
      nm_avg: nmSum / nmCount,
      un_avg: unSum / unCount,
      true_nm_ratio: nmSum > 0 ? (trueSum / trueCount) / (nmSum / nmCount) : Infinity,
    };
  }

  return result;
}

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

  const themeQueryMap = {};
  for (const theme of themes) {
    const indices = [];
    for (let qi = 0; qi < queries.length; qi++) {
      if (queries[qi].theme_id === theme.id) indices.push(qi);
    }
    themeQueryMap[theme.id] = indices;
  }

  console.log("\n=== ANALYSIS ===");
  console.log("EdgeStore stored weight: grows unboundedly (no internal decay)");
  console.log("bonus(): applies effectiveHebbian (decay) then tanh then maxBonus(0.3)");
  console.log("");
  console.log("For the Related: section, we need to know the bonus values at query time.");
  console.log("Queries happen within sessions, so the relevant measurement time is:");
  console.log("  1. 'now' (immediately after last reinforcement, ~minutes apart)");
  console.log("  2. '+6h' (next morning, after overnight gap)");
  console.log("  3. '+24h' (next day, after full day gap)");
  console.log("");

  // === Run the simulation with REALISTIC session gaps ===
  const totalWeeks = 104;
  console.log(`=== SCENARIO: Realistic usage (${totalWeeks} weeks, 2 years) ===`);
  console.log("User sessions: 3-8/week, 12-24h gap between sessions, 2-5 queries/session per theme.");
  console.log();

  const edgeStore = new EdgeStore(null);
  const weeklyStored = [];

  for (let week = 0; week < totalWeeks; week++) {
    let sessionTime = new Date(Date.now() + week * 7 * DAY_SECS * 1000);

    for (const theme of themes) {
      const queryIndices = themeQueryMap[theme.id];
      for (let s = 0; s < 2; s++) {
        const nQueries = 2 + Math.floor(Math.random() * 2);
        for (let q = 0; q < nQueries; q++) {
          sessionTime = new Date(sessionTime.getTime() + (60000 + Math.random() * 300000));
          const qi = queryIndices[(week * 20 + s * 10 + q) % queryIndices.length];
          const query = queries[qi];

          const ranked = records
            .map((r) => ({ id: r.id, sim: cosine(query.embedding, r.embedding) }))
            .sort((a, b) => b.sim - a.sim)
            .slice(0, k);
          const topK = ranked.map((r) => r.id);
          edgeStore.reinforceRecall(topK, []);
        }
        // Session gap: 12-24h
        sessionTime = new Date(sessionTime.getTime() + (12 + Math.random() * 12) * 3600 * 1000);
      }
    }

    // Measure stored weight (no decay)
    let trueTotal = 0, nmTotal = 0, unTotal = 0;
    let trueCount = 0, nmCount = 0, unCount = 0;

    for (const tp of corpus.true_pairs) { trueTotal += edgeStore.weight(tp.a, tp.b); trueCount++; }
    for (const nm of corpus.near_miss_pairs) { nmTotal += edgeStore.weight(nm.a, nm.b); nmCount++; }
    for (const up of corpus.unrelated_pairs) { unTotal += edgeStore.weight(up.a, up.b); unCount++; }

    weeklyStored.push({
      week,
      true_avg: trueTotal / trueCount,
      nm_avg: nmTotal / nmCount,
      un_avg: unTotal / unCount,
      true_nm_ratio: nmTotal > 0 ? (trueTotal / trueCount) / (nmTotal / nmCount) : Infinity,
    });
  }

  console.log("Stored weight convergence (measured at session boundary):");
  for (let i = 0; i < weeklyStored.length; i += 5) {
    const w = weeklyStored[i];
    const rStr = w.true_nm_ratio === Infinity ? "INF" : w.true_nm_ratio.toFixed(2);
    console.log(`  Week ${String(w.week).padStart(4)}: true=${w.true_avg.toFixed(4)} nm=${w.nm_avg.toFixed(4)} un=${w.un_avg.toFixed(4)} ratio=${rStr}x`);
  }

  // === Bonus() at various measurement times ===
  console.log("\n=== BONUS() AT VARIOUS TIMES AFTER LAST SESSION ===");
  console.log("  (All weights at session boundary >> tanh threshold; bonus always 0.3)");
  console.log("  The ratio is meaningful only BEFORE tanh saturation.");
  console.log("  Below shows effectiveHebbian (pre-tanh) at various gaps.\n");
  const offsets = [
    { sec: 0, label: "now" },
    { sec: 6 * 3600, label: "+6h" },
    { sec: 12 * 3600, label: "+12h" },
    { sec: 24 * 3600, label: "+1d" },
    { sec: 48 * 3600, label: "+2d" },
    { sec: 7 * 86400, label: "+7d" },
    { sec: 14 * 86400, label: "+14d" },
    { sec: 30 * 86400, label: "+30d" },
  ];
  for (const o of offsets) {
    let trueSum = 0, nmSum = 0, unSum = 0, tc = 0, nc = 0, uc = 0;
    for (const tp of corpus.true_pairs) {
      const w = edgeStore.weight(tp.a, tp.b);
      trueSum += w * Math.pow(2, -o.sec / HALF_LIFE_SECS);
      tc++;
    }
    for (const nm of corpus.near_miss_pairs) {
      const w = edgeStore.weight(nm.a, nm.b);
      nmSum += w * Math.pow(2, -o.sec / HALF_LIFE_SECS);
      nc++;
    }
    for (const up of corpus.unrelated_pairs) {
      const w = edgeStore.weight(up.a, up.b);
      unSum += w * Math.pow(2, -o.sec / HALF_LIFE_SECS);
      uc++;
    }
    const trueBonus = Math.tanh(trueSum / tc) * 0.3;
    const nmBonus = Math.tanh(nmSum / nc) * 0.3;
    const unBonus = Math.tanh(unSum / uc) * 0.3;
    const ratio = (trueSum / tc) / Math.max(nmSum / nc, 1e-9);
    const rStr = ratio === Infinity ? "INF" : ratio.toFixed(2);
    console.log(`  ${o.label.padEnd(6)}: eff_true=${(trueSum/tc).toFixed(1).padStart(6)} bonus_tanh=${trueBonus.toFixed(4)}  eff_nm=${(nmSum/nc).toFixed(1).padStart(6)} bonus_tanh=${nmBonus.toFixed(4)}  ratio=${rStr}x`);
  }

  // === Per-theme equilibrium ===
  console.log("\n=== PER-THEME EQUILIBRIUM (stored weight at session boundary) ===");
  console.log("  Theme                      | True Bonus  | Near-miss | Ratio");
  console.log("  -------------------------- | ----------- | --------- | ------");

  const perThemeResults = {};
  for (const theme of themes) {
    const themeMembers = corpus.records.filter((r) => r.theme_id === theme.id && r.role === "member");
    const nearMisses = corpus.records.filter((r) => r.theme_id === theme.id && r.role === "near-miss");

    let trueTotal = 0, nmTotal = 0, trueCount = 0, nmCount = 0;

    for (let i = 0; i < themeMembers.length; i++) {
      for (let j = i + 1; j < themeMembers.length; j++) {
        trueTotal += edgeStore.weight(themeMembers[i].id, themeMembers[j].id);
        trueCount++;
      }
    }
    for (let i = 0; i < themeMembers.length; i++) {
      for (const nm of nearMisses) {
        nmTotal += edgeStore.weight(themeMembers[i].id, nm.id);
        nmCount++;
      }
    }

    const trueAvg = trueCount > 0 ? trueTotal / trueCount : 0;
    const nmAvg = nmCount > 0 ? nmTotal / nmCount : 0;
    const ratio = nmAvg > 0 ? trueAvg / nmAvg : (trueAvg > 0 ? Infinity : 0);
    const ratioStr = ratio === Infinity ? "  INF" : ` ${(ratio).toFixed(2)}x`;

    // Apply tanh + maxBonus
    const trueBonus = Math.tanh(trueAvg) * 0.3;
    const nmBonus = Math.tanh(nmAvg) * 0.3;

    console.log(`  ${theme.name.padEnd(26)} | ${(trueBonus).toFixed(4).padStart(9)} | ${(nmBonus).toFixed(4).padStart(9)} |${ratioStr}`);

    perThemeResults[theme.id] = {
      name: theme.name,
      true_bonus: trueBonus,
      nm_bonus: nmBonus,
      true_nm_ratio: ratio,
    };
  }

  // Save results
  const results = {
    version: corpus.version,
    k,
    n_records: allRecords.length,
    n_haystack: HAYSTACK_SIZE,
    embed_model: EMBED_MODEL,
    half_life_days: 7,
    alpha: 0.1,
    max_bonus: 0.3,
    total_weeks: totalWeeks,
    scenarios: {
      stored_weight: {
        label: "Stored weight at session boundary (unbounded growth)",
        weekly: weeklyStored,
      },
      bonus_at_offsets: {
        label: "bonus() at various times after last session (tanh-decayed)",
        offsets: offsets.map(dt => ({
          time: dt === 0 ? "now" : dt <= 86400 ? `+${(dt / 3600).toFixed(1)}h` : `+${(dt / 86400).toFixed(1)}d`,
          measurements: measureEffectiveAtOffset(edgeStore, [dt], corpus.true_pairs, corpus.near_miss_pairs, corpus.unrelated_pairs)[dt === 0 ? "now" : dt <= 86400 ? `+${dt / 3600}h` : `+${dt / 86400}d`],
          effectiveHebbian: measureEffectiveAtOffset(edgeStore, [dt], corpus.true_pairs, corpus.near_miss_pairs, corpus.unrelated_pairs)[dt === 0 ? "now" : dt <= 86400 ? `+${dt / 3600}h` : `+${dt / 86400}d`],
          tanh_bonus: {
            true: Math.tanh(weeklyStored[weeklyStored.length - 1].true_avg) * 0.3,
            nm: Math.tanh(weeklyStored[weeklyStored.length - 1].nm_avg) * 0.3,
            un: Math.tanh(weeklyStored[weeklyStored.length - 1].un_avg) * 0.3,
          },
        })),
      },
      per_theme: perThemeResults,
    },
  };

  fs.writeFileSync(RESULTS_PATH, JSON.stringify(results, null, 2));
  console.log(`\nResults saved to ${RESULTS_PATH}`);

  // Final analysis
  console.log("\n=== FINAL ANALYSIS ===");

  // Get the last stored weight measurement
  const last = weeklyStored[weeklyStored.length - 1];
  const trueTanh = Math.tanh(last.true_avg) * 0.3;
  const nmTanh = Math.tanh(last.nm_avg) * 0.3;
  const unTanh = Math.tanh(last.un_avg) * 0.3;

  console.log(`At session boundary (2 years of use):`);
  console.log(`  True pair stored weight: ${last.true_avg.toFixed(2)} → bonus(tanh): ${trueTanh.toFixed(4)} (saturated: ${trueTanh >= 0.299})`);
  console.log(`  Near-miss stored weight: ${last.nm_avg.toFixed(2)} → bonus(tanh): ${nmTanh.toFixed(4)} (saturated: ${nmTanh >= 0.299})`);
  console.log(`  Unrelated stored weight: ${last.un_avg.toFixed(2)} → bonus(tanh): ${unTanh.toFixed(4)}`);
  console.log(`  True/Near-miss ratio (stored): ${(last.true_avg / last.nm_avg).toFixed(2)}x`);
  console.log(`  True/Near-miss ratio (bonus): ${(trueTanh / Math.max(nmTanh, 1e-9)).toFixed(2)}x`);
  console.log();
  console.log(`After 1 week of decay:`);
  const weekDecay = 7 * DAY_SECS;
  const trueWDec = last.true_avg * Math.pow(2, -weekDecay / HALF_LIFE_SECS);
  const nmWDec = last.nm_avg * Math.pow(2, -weekDecay / HALF_LIFE_SECS);
  const trueBD = Math.tanh(trueWDec) * 0.3;
  const nmBD = Math.tanh(nmWDec) * 0.3;
  console.log(`  True pair: ${trueWDec.toFixed(2)} → bonus(tanh): ${trueBD.toFixed(4)}`);
  console.log(`  Near-miss: ${nmWDec.toFixed(2)} → bonus(tanh): ${nmBD.toFixed(4)}`);
  console.log(`  Ratio: ${(trueWDec / Math.max(nmWDec, 1e-9)).toFixed(2)}x`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
