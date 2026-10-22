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
 * Quick diagnostic: same corpus, 500 haystack records (heavier competitive
 * pressure). Tests whether the true/near-miss gap widens or stays flat.
 */
"use strict";

const fs = require("fs");
const path = require("path");
const { EdgeStore } = require("../../edges.js");
const { cosine } = require("../pipeline.js");

const CORPUS_PATH = path.join(__dirname, "../corpora/fire-together-corpus.json");

async function embedBatch(texts) {
  const body = JSON.stringify({ model: process.env.EMBED_MODEL || "text-embedding-nomic-embed-text-v1.5", input: texts });
  const buf = Buffer.from(body);
  const res = await fetch("http://localhost:1234/v1/embeddings", {
    method: "POST",
    headers: { "Content-Type": "application/json", "Content-Length": String(buf.length) },
    body: buf,
  });
  if (!res.ok) throw new Error(`Embed failed: ${res.status}`);
  const json = JSON.parse(await res.text());
  return json.data.map((d) => d.embedding);
}

function generateHaystackRecords(count) {
  const templates = [
    () => `I bought a new ${["blue","red","green","black","white","gray","navy","brown"][Math.floor(Math.random()*8)]} ${["jacket","shirt","pants","dress","sweater","coat","hoodie","scarf","hat","belt"][Math.floor(Math.random()*10)]} from ${["Zara","H&M","Uniqlo","Target","Walmart","Amazon","Nordstrom","Old Navy","Gap","Costco"][Math.floor(Math.random()*10)]} last week`,
    () => `I've been cooking ${["Thai","Italian","Mexican","Japanese","Indian","Korean","French","Chinese","Vietnamese","Lebanese"][Math.floor(Math.random()*10)]} food at home more often lately`,
    () => `My ${["dad","mom","brother","sister","cousin","uncle","aunt","neighbor","friend","colleague"][Math.floor(Math.random()*10)]} lives in ${["Denver","Austin","Portland","Seattle","Chicago","Boston","Miami","Atlanta","Nashville","Minneapolis"][Math.floor(Math.random()*10)]}`,
    () => `I need to ${["update my resume","schedule a dentist appointment","renew my passport","pay the electric bill","change my oil","fix the leaky faucet","replace the brake pads","clean the gutters","update my insurance","file my taxes"][Math.floor(Math.random()*10)]}`,
    () => `I read a great book about ${["neuroscience","history","cooking","travel","philosophy","psychology","astronomy","genetics","art","politics"][Math.floor(Math.random()*10)]} this past weekend`,
    () => `The ${["parking garage","grocery store","coffee shop","library","gym","hospital","bank","post office","hardware store","pharmacy"][Math.floor(Math.random()*10)]} on ${["Main","Oak","Elm","Maple","Cedar","Pine","Birch","Walnut","Chestnut","Willow"][Math.floor(Math.random()*10)]} Street is always crowded`,
    () => `I'm trying to cut back on ${["sugar","caffeine","social media","screen time","carbs","dairy","processed food","meat","alcohol","junk food"][Math.floor(Math.random()*10)]}`,
    () => `We had ${["pizza","sushi","tacos","ramen","burgers","pasta","curry","steak","bbq","falafel"]} for dinner last night`,
    () => `My ${["dog","cat","hamster","fish","bird","rabbit","turtle","guinea pig","parrot","ferret"][Math.floor(Math.random()*10)]} ${["loves","hates","enjoys","sleeps in","chases","chews on","perches on","bites","grooms","follows around"]} ${["the couch","the door","the garden","the kitchen","the backyard","the basement","the mailbox","the window","the laundry","the garden hose"]}`,
    () => `I went ${["hiking","swimming","biking","running","kayaking","rock climbing","surfing","fishing","camping","skiing"]} last weekend`,
    () => `I've been listening to a lot of ${["podcasts","lo-fi","jazz","classical","rock","hip-hop","indie","folk","electronic","country"]}`,
    () => `I ${["love","hate","need","want","miss","enjoy","prefer","avoid","appreciate","dislike"]} ${["morning coffee","late nights","early mornings","weekends","cold weather","hot weather","rain","sunshine","snow","wind"]}`,
    () => `The ${["new","old","local","famous","hidden","chain","independent","corner","downtown","suburban"]}${["restaurant","bar","cafe","bookstore","store","theater","stadium","museum","gallery","studio"]} is ${["closed","renovating","hiring","expanding","closing down","opening soon","under new management","fully booked","under review","doing well"]}`,
    () => `I should really ${["exercise more","eat better","sleep more","save money","travel more","learn a new skill","call my mom","clean the house","declutter","start meditating"]}`,
    () => `My ${["boss","teacher","professor","manager","coach","supervisor","mentor","advisor","instructor","trainer"]} said I need to ${["improve my communication","work on my patience","be more organized","take more initiative","delegate better","manage my time","be more confident","speak up more","collaborate better","follow through"]}`,
    () => `I went to ${["the beach","the mountains","the lake","the desert","the forest","the canyon","the waterfall","the coastline","the island","the river"]} and the ${["view","weather","water","air","scenery","nature","wildlife","sunrise","sunset","stars"]} was incredible`,
    () => `I bought a ${["laptop","phone","tablet","monitor","keyboard","headphones","camera","drone","watch","speaker"]} from ${["Apple","Samsung","Dell","Sony","Bose","Canon","DJI","Garmin","JBL","Anker"]}`,
    () => `The ${["traffic","construction","road work","accident","detour","lane closure","bridge repair","utility work","street fair","parade"]} made me ${["late","take a detour","wait an hour","find an alternate route","lose my patience","call in late","get stuck","turn around","replan","sit in traffic"]}`,
    () => `I've been ${["working out","going to the gym","training for a race","doing yoga","practicing martial arts","learning to dance","stretching more","lifting weights","running marathons","swimming laps"]}`,
    () => `I need to ${["return this","exchange this","get a refund","file a warranty claim","complain about","report","upgrade","renew","cancel","reactivate"]}${[" my subscription","the warranty","the contract","the order","the app","the service","the plan","the membership","the policy","the account"]}`,
  ];
  const records = [];
  for (let i = 0; i < count; i++) {
    const base = templates[i % templates.length]();
    const noise = ["actually","really","just","sort of","kind of","pretty much","totally","basically","honestly","frankly"];
    const nw = noise[Math.floor(Math.random() * noise.length)];
    const text = Math.random() < 0.5 ? `${nw.charAt(0).toUpperCase() + nw.slice(1)} ${base}` : `${base} ${nw}.`;
    records.push({ id: `h500_${i}`, text, theme_id: null, role: "distractor" });
  }
  return records;
}

function pairKey(a, b) {
  const x = String(a), y = String(b);
  return x < y ? x + "\t" + y : y + "\t" + x;
}

async function main() {
  console.log("Loading corpus...");
  const corpus = JSON.parse(fs.readFileSync(CORPUS_PATH, "utf8"));
  const nearMissPairSet = new Set(corpus.near_miss_pairs.map((p) => pairKey(p.a, p.b)));
  const truePairSet = new Set(corpus.true_pairs.map((p) => pairKey(p.a, p.b)));

  const haystackCount = 500;
  console.log(`Generating ${haystackCount} haystack records...`);
  const haystack = generateHaystackRecords(haystackCount);

  const allRecords = [...corpus.records, ...haystack];
  const allTexts = [...allRecords.map((r) => r.text), ...corpus.queries.map((q) => q.query)];
  const total = allTexts.length;

  console.log(`Embedding ${total} texts in batches of 10...`);
  const allEmb = [];
  for (let i = 0; i < total; i += 10) {
    console.log(`  batch ${Math.floor(i/10)+1}/${Math.ceil(total/10)}`);
    const batch = await embedBatch(allTexts.slice(i, i + 10));
    allEmb.push(...batch);
  }

  let off = 0;
  const records = allRecords.map((r) => ({ ...r, embedding: allEmb[off++] }));
  const queries = corpus.queries.map((q) => ({ ...q, embedding: allEmb[off++] }));

  const nQueries = queries.length;
  const k = 5;
  const cofireCounts = {};
  const edgeStore = new EdgeStore(null);
  let h1Hits = 0;

  for (let qi = 0; qi < queries.length; qi++) {
    const ranked = records.map((r) => ({
      id: r.id, text: r.text, sim: cosine(queries[qi].embedding, r.embedding),
      role: r.role, theme_id: r.theme_id,
    })).sort((a, b) => b.sim - a.sim).slice(0, k);

    const pids = ranked.map((r) => r.id);

    if (queries[qi].target_id) {
      if (pids.includes(queries[qi].target_id)) h1Hits++;
    } else if (queries[qi].theme_id) {
      const members = records.filter((r) => r.theme_id === queries[qi].theme_id && r.role === "member").map((r) => r.id);
      if (pids.some((id) => members.includes(id))) h1Hits++;
    }

    for (let i = 0; i < pids.length; i++) {
      for (let j = i + 1; j < pids.length; j++) {
        const key = pairKey(pids[i], pids[j]);
        cofireCounts[key] = (cofireCounts[key] || 0) + 1;
      }
    }
    edgeStore.reinforceRecall(pids, []);
  }

  // Metrics
  const h1Rate = h1Hits / nQueries;
  const truePairRates = corpus.true_pairs.map((p) => {
    const key = pairKey(p.a, p.b);
    const count = cofireCounts[key] || 0;
    return { pair: `${p.a} ↔ ${p.b}`, theme_id: p.theme_id, rate: count / nQueries };
  });
  const trueMeanRate = truePairRates.reduce((s, r) => s + r.rate, 0) / truePairRates.length;

  const nmRates = corpus.near_miss_pairs.map((p) => {
    const key = pairKey(p.a, p.b);
    return { pair: `${p.a} ↔ ${p.b}`, theme_id: p.theme_id, rate: (cofireCounts[key] || 0) / nQueries };
  });
  const nmMeanRate = nmRates.reduce((s, r) => s + r.rate, 0) / nmRates.length;
  const nmMaxRate = Math.max(...nmRates.map((r) => r.rate));

  const unRates = corpus.unrelated_pairs.map((p) => {
    const key = pairKey(p.a, p.b);
    return (cofireCounts[key] || 0) / nQueries;
  });
  const unMeanRate = unRates.reduce((s, r) => s + r.rate, 0) / unRates.length;

  console.log("\n=== RESULTS (500 haystack) ===");
  console.log(`n_records: ${allRecords.length}`);
  console.log(`n_queries: ${nQueries}`);
  console.log(`H1: ${(h1Rate * 100).toFixed(1)}%`);
  console.log(`H2 true mean: ${(trueMeanRate * 100).toFixed(2)}%`);
  console.log(`H3 near-miss mean: ${(nmMeanRate * 100).toFixed(2)}%  max: ${(nmMaxRate * 100).toFixed(2)}%`);
  console.log(`H4 unrelated mean: ${(unMeanRate * 100).toFixed(2)}%`);
  console.log(`True/Near ratio: ${(trueMeanRate / (nmMeanRate || 0.001)).toFixed(2)}x`);
  console.log(`Delta: ${((trueMeanRate - nmMeanRate) * 100).toFixed(2)}%`);

  // Per-theme
  console.log("\nPer-theme true max vs near-miss max:");
  for (const theme of corpus.themes) {
    const tids = corpus.true_pairs.filter((p) => p.theme_id === theme.id);
    const nms = corpus.near_miss_pairs.filter((p) => p.theme_id === theme.id);
    let tmax = 0, nmax = 0;
    for (const p of tids) tmax = Math.max(tmax, (cofireCounts[pairKey(p.a, p.b)] || 0) / nQueries);
    for (const p of nms) nmax = Math.max(nmax, (cofireCounts[pairKey(p.a, p.b)] || 0) / nQueries);
    console.log(`  ${theme.name.padEnd(30)} true=${(tmax*100).toFixed(1)}%  nm=${(nmax*100).toFixed(1)}%  ratio=${(tmax/(nmax||0.001)).toFixed(2)}x`);
  }
}

main().catch(console.error);
