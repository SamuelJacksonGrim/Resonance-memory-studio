/*
 * eval/substrate/generate-fire-together.js — 4.0b Hermes fire-together corpus generator.
 *
 * Implements RFC 0011 §7.2:
 * - 5 themes × 3 members = 15 on-theme facts
 * - 5 near-misses (1 per theme)
 * - 5 unrelated distractors
 * - ~500 total phrasings (member paraphrases, theme-level queries, near-miss queries, distractors)
 *
 * Generates natural conversational phrasings using Gemini 3.8 Flash,
 * computes 768-dim embeddings via gemini-embedding-001 batchEmbedContents,
 * and saves a self-contained, deterministic fixture file.
 */

"use strict";

const fs = require("fs");
const path = require("path");

const CORPUS_PATH = path.join(__dirname, "../corpora/fire-together-corpus.json");

const THEMES = [
  {
    id: "morning-drink",
    name: "Morning Drink / Cortado",
    members: [
      { id: "t1_m1", text: "I drink an oat-milk cortado every morning" },
      { id: "t1_m2", text: "My morning coffee is a double cortado with oat milk and no sugar" },
      { id: "t1_m3", text: "I brew a cortado with steamed oat milk right before starting work" },
    ],
    near_miss: {
      id: "t1_nm",
      text: "I drink hot chamomile tea with honey at night before bed",
      relation: "drink/evening routine vs morning coffee",
    },
  },
  {
    id: "climbing",
    name: "Indoor Bouldering",
    members: [
      { id: "t2_m1", text: "I climb at the south bouldering gym on Tuesday evenings" },
      { id: "t2_m2", text: "Tuesday nights I go indoor bouldering at the south gym" },
      { id: "t2_m3", text: "The south climbing gym is where I work on my bouldering projects" },
    ],
    near_miss: {
      id: "t2_nm",
      text: "I lift free weights at the downtown fitness center on Thursday mornings",
      relation: "downtown weightlifting vs south climbing gym",
    },
  },
  {
    id: "sister-naima",
    name: "Sister Naima",
    members: [
      { id: "t3_m1", text: "My sister Naima teaches chemistry at the local high school" },
      { id: "t3_m2", text: "Naima is my sister and she works as a high school chemistry teacher" },
      { id: "t3_m3", text: "I call my sister Naima every Sunday evening to catch up" },
    ],
    near_miss: {
      id: "t3_nm",
      text: "My coworker Naima is a frontend engineer on the infrastructure team",
      relation: "coworker Naima vs sister Naima",
    },
  },
  {
    id: "bookshelf",
    name: "Walnut Bookshelf Project",
    members: [
      { id: "t4_m1", text: "I am building a custom walnut bookshelf in my garage workshop" },
      { id: "t4_m2", text: "The walnut bookcase in my garage is six feet tall with four shelves" },
      { id: "t4_m3", text: "I am applying oil finish to my handmade walnut bookcase" },
    ],
    near_miss: {
      id: "t4_nm",
      text: "I bought a flat-pack pine desk from IKEA for my study",
      relation: "bought IKEA pine desk vs built walnut bookcase",
    },
  },
  {
    id: "allergy",
    name: "Penicillin Allergy",
    members: [
      { id: "t5_m1", text: "I have a severe allergic reaction to penicillin and amoxicillin" },
      { id: "t5_m2", text: "My medical chart notes an anaphylactic allergy to penicillin antibiotics" },
      { id: "t5_m3", text: "I wear a medical alert bracelet for my penicillin allergy" },
    ],
    near_miss: {
      id: "t5_nm",
      text: "I take daily vitamin D and zinc supplements with breakfast",
      relation: "daily supplements vs antibiotic allergy",
    },
  },
];

const DISTRACTORS = [
  { id: "d1", text: "The capital of Oregon is Salem, which was incorporated in 1857" },
  { id: "d2", text: "My car is a blue 2018 Honda Civic with manual transmission" },
  { id: "d3", text: "I renewed my passport at the downtown post office last September" },
  { id: "d4", text: "Apollo 11 landed on the lunar surface on July 20, 1969" },
  { id: "d5", text: "I replaced the furnace filter with an allergen-rated pleated filter" },
];

function cleanJson(raw) {
  if (!raw) return [];
  let s = raw.trim();
  if (s.startsWith("```")) {
    s = s.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "");
  }
  try {
    return JSON.parse(s);
  } catch (err) {
    const match = s.match(/\[[\s\S]*\]/);
    if (match) return JSON.parse(match[0]);
    throw err;
  }
}

async function callGemini(prompt, systemInstruction = "") {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error("GEMINI_API_KEY not set");
  let lastErr = null;

  for (let attempt = 0; attempt < 5; attempt++) {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key=${apiKey}`;
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          contents: [{ parts: [{ text: prompt }] }],
          systemInstruction: systemInstruction ? { parts: [{ text: systemInstruction }] } : undefined,
          generationConfig: {
            temperature: 0.7,
            responseMimeType: "application/json",
            thinkingConfig: { thinkingLevel: "low" },
          },
        }),
      });
      if (!res.ok) {
        lastErr = new Error(`Gemini generateContent error ${res.status}: ${await res.text()}`);
        await new Promise((r) => setTimeout(r, 1500 * (attempt + 1)));
        continue;
      }
      const data = await res.json();
      const rawText = data.candidates?.[0]?.content?.parts?.[0]?.text;
      return cleanJson(rawText);
    } catch (err) {
      lastErr = err;
      await new Promise((r) => setTimeout(r, 1500 * (attempt + 1)));
    }
  }
  throw lastErr;
}

async function batchEmbed(texts) {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error("GEMINI_API_KEY not set");
  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents?key=${apiKey}`;
  const BATCH_SIZE = 50;
  const embeddings = [];

  for (let i = 0; i < texts.length; i += BATCH_SIZE) {
    const chunk = texts.slice(i, i + BATCH_SIZE);
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        requests: chunk.map((text) => ({
          model: "models/gemini-embedding-001",
          content: { parts: [{ text }] },
          outputDimensionality: 768,
        })),
      }),
    });
    if (!res.ok) throw new Error(`batchEmbedContents error ${res.status}: ${await res.text()}`);
    const data = await res.json();
    for (const e of data.embeddings) {
      embeddings.push(e.values);
    }
  }
  return embeddings;
}

const PROGRESS_PATH = path.join(__dirname, "generate-progress.json");

async function generateAllPhrasings() {
  console.log("Generating natural phrasings via Gemini...");
  let progress = {};
  if (fs.existsSync(PROGRESS_PATH)) {
    try { progress = JSON.parse(fs.readFileSync(PROGRESS_PATH, "utf8")); } catch {}
  }

  const queries = [];

  // 1. For each member (15 facts): generate 20 natural paraphrases
  for (const theme of THEMES) {
    for (const member of theme.members) {
      const key = `member_${member.id}`;
      let phrasings = progress[key];
      if (!phrasings) {
        console.log(`  Generating paraphrases for ${member.id}: "${member.text}"...`);
        const prompt = `Given this personal memory fact: "${member.text}".
Generate exactly 20 distinct, realistic, natural conversational phrasings that an assistant user might say or ask when referring to, querying, or recalling this fact.
Include:
- direct statements / reminders ("remember that I drink...", "just so you know, ...")
- direct questions ("what do I drink...?", "how do I take my...?")
- casual/colloquial shorthand ("cortado with oat milk for me", "my standard morning cup")
- contextual inquiries ("what's my morning routine?", "did I already have coffee?")
Return a JSON array of 20 strings. Example: ["phrasing 1", "phrasing 2", ...]`;
        phrasings = await callGemini(prompt);
        progress[key] = phrasings;
        fs.writeFileSync(PROGRESS_PATH, JSON.stringify(progress, null, 2));
      } else {
        console.log(`  Using cached paraphrases for ${member.id}`);
      }
      for (const p of phrasings.slice(0, 20)) {
        queries.push({
          query: p,
          kind: "member-paraphrase",
          target_id: member.id,
          theme_id: theme.id,
        });
      }
    }

    // 2. For each theme: generate 20 theme-level queries
    const themeKey = `theme_${theme.id}`;
    let themeQueries = progress[themeKey];
    if (!themeQueries) {
      console.log(`  Generating theme-level queries for theme "${theme.name}"...`);
      const promptTheme = `Given these related personal memory facts in theme "${theme.name}":
${theme.members.map((m) => `- "${m.text}"`).join("\n")}
Generate exactly 20 natural conversational questions or statements that broadly address or touch upon this theme as a whole.
Users might ask open questions like "tell me about my morning drinks", "what's my exercise routine", etc.
Return a JSON array of 20 strings. Example: ["query 1", "query 2", ...]`;
      themeQueries = await callGemini(promptTheme);
      progress[themeKey] = themeQueries;
      fs.writeFileSync(PROGRESS_PATH, JSON.stringify(progress, null, 2));
    } else {
      console.log(`  Using cached queries for theme "${theme.name}"`);
    }
    for (const p of themeQueries.slice(0, 20)) {
      queries.push({
        query: p,
        kind: "theme-query",
        target_id: null,
        theme_id: theme.id,
      });
    }

    // 3. For each near-miss: generate 15 phrasings
    const nmKey = `nearmiss_${theme.near_miss.id}`;
    let nmQueries = progress[nmKey];
    if (!nmQueries) {
      console.log(`  Generating near-miss phrasings for ${theme.near_miss.id}: "${theme.near_miss.text}"...`);
      const promptNM = `Given this near-miss fact: "${theme.near_miss.text}" (which contrasts with the theme "${theme.name}").
Generate exactly 15 distinct, realistic, natural phrasings or queries about this specific fact.
Return a JSON array of 15 strings. Example: ["query 1", "query 2", ...]`;
      nmQueries = await callGemini(promptNM);
      progress[nmKey] = nmQueries;
      fs.writeFileSync(PROGRESS_PATH, JSON.stringify(progress, null, 2));
    } else {
      console.log(`  Using cached near-miss for ${theme.near_miss.id}`);
    }
    for (const p of nmQueries.slice(0, 15)) {
      queries.push({
        query: p,
        kind: "near-miss-query",
        target_id: theme.near_miss.id,
        theme_id: theme.id,
      });
    }
  }

  // 4. For each distractor: generate 10 phrasings
  for (const dist of DISTRACTORS) {
    const distKey = `distractor_${dist.id}`;
    let distQueries = progress[distKey];
    if (!distQueries) {
      console.log(`  Generating distractor phrasings for ${dist.id}: "${dist.text}"...`);
      const promptDist = `Given this unrelated distractor fact: "${dist.text}".
Generate exactly 10 distinct questions or statements about this fact.
Return a JSON array of 10 strings.`;
      distQueries = await callGemini(promptDist);
      progress[distKey] = distQueries;
      fs.writeFileSync(PROGRESS_PATH, JSON.stringify(progress, null, 2));
    } else {
      console.log(`  Using cached distractor for ${dist.id}`);
    }
    for (const p of distQueries.slice(0, 10)) {
      queries.push({
        query: p,
        kind: "distractor-query",
        target_id: dist.id,
        theme_id: null,
      });
    }
  }

  console.log(`Total queries generated: ${queries.length}`);
  return queries;
}

async function main() {
  const allRecords = [];
  for (const theme of THEMES) {
    for (const m of theme.members) {
      allRecords.push({ id: m.id, text: m.text, theme_id: theme.id, role: "member" });
    }
    allRecords.push({ id: theme.near_miss.id, text: theme.near_miss.text, theme_id: theme.id, role: "near-miss" });
  }
  for (const d of DISTRACTORS) {
    allRecords.push({ id: d.id, text: d.text, theme_id: null, role: "distractor" });
  }

  const queries = await generateAllPhrasings();

  console.log("Embedding 25 facts + queries...");
  const recordTexts = allRecords.map((r) => r.text);
  const queryTexts = queries.map((q) => q.query);
  const allTexts = [...recordTexts, ...queryTexts];

  const embeddings = await batchEmbed(allTexts);

  for (let i = 0; i < allRecords.length; i++) {
    allRecords[i].embedding = embeddings[i];
  }
  for (let i = 0; i < queries.length; i++) {
    queries[i].embedding = embeddings[allRecords.length + i];
  }

  // Define true theme pairs
  const true_pairs = [];
  for (const theme of THEMES) {
    const ids = theme.members.map((m) => m.id);
    for (let i = 0; i < ids.length; i++) {
      for (let j = i + 1; j < ids.length; j++) {
        true_pairs.push({ a: ids[i], b: ids[j], theme_id: theme.id });
      }
    }
  }

  // Define near-miss pairs (each member of the theme paired with the near-miss)
  const near_miss_pairs = [];
  for (const theme of THEMES) {
    const nmId = theme.near_miss.id;
    for (const m of theme.members) {
      near_miss_pairs.push({ a: m.id, b: nmId, theme_id: theme.id });
    }
  }

  // Define unrelated pairs (pairs across different themes or with distractors, excluding near-miss)
  const unrelated_pairs = [];
  const allIds = allRecords.map((r) => r.id);
  const themeMap = new Map(allRecords.map((r) => [r.id, r.theme_id]));
  const roleMap = new Map(allRecords.map((r) => [r.id, r.role]));

  for (let i = 0; i < allIds.length; i++) {
    for (let j = i + 1; j < allIds.length; j++) {
      const a = allIds[i], b = allIds[j];
      const tA = themeMap.get(a), tB = themeMap.get(b);
      // If they belong to different themes (or at least one has no theme), and neither is a near-miss of the other's theme
      if (tA === null || tB === null || tA !== tB) {
        unrelated_pairs.push({ a, b });
      }
    }
  }

  const corpus = {
    version: "4.0b-hermes-v1",
    created_at: new Date().toISOString(),
    embedding_model: "models/gemini-embedding-001",
    embedding_dim: 768,
    themes: THEMES.map((t) => ({ id: t.id, name: t.name })),
    records: allRecords,
    queries,
    true_pairs,
    near_miss_pairs,
    unrelated_pairs,
  };

  fs.mkdirSync(path.dirname(CORPUS_PATH), { recursive: true });
  fs.writeFileSync(CORPUS_PATH, JSON.stringify(corpus, null, 2), "utf8");
  console.log(`Corpus written successfully to ${CORPUS_PATH}!`);
  console.log(`  Records: ${corpus.records.length}`);
  console.log(`  Queries: ${corpus.queries.length}`);
  console.log(`  True pairs: ${corpus.true_pairs.length}`);
  console.log(`  Near-miss pairs: ${corpus.near_miss_pairs.length}`);
  console.log(`  Unrelated pairs: ${corpus.unrelated_pairs.length}`);
}

if (require.main === module) {
  main().catch((err) => {
    console.error("Fatal error:", err);
    process.exit(1);
  });
}

module.exports = { main, THEMES, DISTRACTORS };
