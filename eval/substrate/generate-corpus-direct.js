/*
 * eval/substrate/generate-corpus-direct.js — 4.0b Hermes fire-together corpus generator.
 *
 * Implements RFC 0011 §7.2:
 * - 5 themes × 3 members = 15 on-theme facts
 * - 5 near-misses (1 per theme)
 * - 5 unrelated distractors
 * - 20 phrasings per member fact (300)
 * - 20 theme-level queries per theme (100)
 * - 15 phrasings per near-miss (75)
 * - 10 phrasings per distractor (50)
 * Total: 25 facts + 525 phrasings = 550 texts
 *
 * Uses gemini-embedding-001 (768 dimensions) via batchEmbedContents.
 * Saves eval/corpora/fire-together-corpus.json.
 */

"use strict";

const fs = require("fs");
const path = require("path");

const CORPUS_PATH = path.join(__dirname, "../corpora/fire-together-corpus.json");

const THEMES_DATA = [
  {
    id: "morning-drink",
    name: "Morning Drink / Cortado",
    members: [
      {
        id: "t1_m1",
        text: "I drink an oat-milk cortado every morning",
        phrasings: [
          "what do I drink in the morning?",
          "what is my morning drink?",
          "I drink an oat milk cortado each morning",
          "my daily morning drink is an oat-milk cortado",
          "remind me what kind of coffee I drink when I wake up",
          "cortado with oat milk every single morning",
          "every day starts with an oat milk cortado for me",
          "do I drink oat-milk cortados in the morning?",
          "what do I have to drink first thing after waking up?",
          "I start every morning with an oat-milk cortado",
          "cortado with oat milk is my daily wake-up beverage",
          "what's that oat milk espresso drink I have every morning?",
          "my morning routine always includes an oat milk cortado",
          "I always make an oat-milk cortado in the morning",
          "what coffee drink do I make every morning?",
          "tell me about my morning cortado habit",
          "oat-milk cortado, that's what I drink every morning",
          "what do I drink every day before starting my tasks?",
          "do I drink an oat milk cortado every day?",
          "my go-to morning beverage is an oat-milk cortado"
        ]
      },
      {
        id: "t1_m2",
        text: "My morning coffee is a double cortado with oat milk and no sugar",
        phrasings: [
          "how do I take my morning coffee?",
          "how do I prepare my morning cortado?",
          "do I put sugar in my morning coffee?",
          "double cortado with oat milk and zero sugar in the morning",
          "how many shots of espresso go in my morning cortado?",
          "my morning coffee recipe: double shot cortado, oat milk, unsweetened",
          "what are the exact specs of my morning coffee?",
          "remind me: do I add any sweetener or sugar to my cortado?",
          "I take my morning coffee as a double cortado with oat milk, no sugar",
          "double shot oat-milk cortado with no sugar is my morning order",
          "do I drink single or double cortado in the mornings?",
          "unsweetened double cortado with oat milk in the morning",
          "what kind of milk and how much sugar in my morning coffee?",
          "my coffee in the morning is double cortado, oat milk, sugar-free",
          "confirm how I like my morning coffee made",
          "no sugar in my morning double oat cortado, right?",
          "double cortado oat milk no sugar please",
          "how much espresso is in my morning brew?",
          "does my morning coffee have oat milk and no sugar?",
          "tell me the details of my morning coffee preference"
        ]
      },
      {
        id: "t1_m3",
        text: "I brew a cortado with steamed oat milk right before starting work",
        phrasings: [
          "what do I brew right before I start working?",
          "I steam oat milk for a cortado just before my workday starts",
          "brewing my cortado right before logging on for work",
          "what's my ritual right before sitting down to work?",
          "do I make a cortado before work?",
          "steaming oat milk for a cortado before clocking in",
          "what beverage do I brew right before starting my shift?",
          "I make a cortado with hot steamed oat milk just before work",
          "right before work I always brew a cortado with oat milk",
          "my pre-work routine involves brewing an oat-milk cortado",
          "do I steam oat milk for coffee before starting work in the morning?",
          "freshly brewed cortado with steamed oat milk right before work begins",
          "what do I prepare to drink before opening my work laptop?",
          "brewing an oat milk cortado before getting to work",
          "I brew a fresh cortado right before work",
          "is it a cortado with steamed oat milk that I make before working?",
          "my pre-work coffee routine: brewing a cortado with steamed oat milk",
          "what coffee do I steam and brew before starting my workday?",
          "remind me of my pre-work brewing routine",
          "steamed oat milk cortado brewed right before starting work"
        ]
      }
    ],
    theme_queries: [
      "what are all my morning coffee habits?",
      "tell me everything about how I drink coffee in the mornings",
      "what is my morning caffeine routine?",
      "tell me about my morning cortado and how I make it",
      "what drinks do I consume in the morning around work?",
      "summarize my morning coffee preferences and preparation",
      "how do I handle my morning coffee before work?",
      "what should I buy for my morning coffee setup?",
      "remind me of my morning coffee routine and ingredients",
      "do I drink cortados with oat milk?",
      "what kind of coffee and milk do I keep at home for mornings?",
      "tell me about my morning drink rituals",
      "what's my morning coffee recipe and schedule?",
      "how do I take coffee when starting the day?",
      "what morning espresso drinks do I enjoy?",
      "give me an overview of my morning cortado habits",
      "everything you know about my morning coffee",
      "when and how do I drink coffee in the morning?",
      "what do I drink before work and in the morning?",
      "describe my morning coffee ritual"
    ],
    near_miss: {
      id: "t1_nm",
      text: "I drink hot chamomile tea with honey at night before bed",
      phrasings: [
        "what do I drink at night before going to bed?",
        "do I drink chamomile tea before sleeping?",
        "hot chamomile tea with honey before bed",
        "what hot beverage do I have at night?",
        "my bedtime drink is chamomile tea with honey",
        "do I put honey in my evening tea?",
        "what kind of tea do I drink at night?",
        "remind me of my nighttime tea ritual",
        "I drink a cup of chamomile tea before going to sleep",
        "chamomile and honey before bed at night",
        "what do I sip before going to sleep?",
        "hot tea with honey at bedtime",
        "do I drink tea at night?",
        "what warm drink helps me unwind before bed?",
        "chamomile tea routine at night"
      ]
    }
  },
  {
    id: "climbing",
    name: "Indoor Bouldering",
    members: [
      {
        id: "t2_m1",
        text: "I climb at the south bouldering gym on Tuesday evenings",
        phrasings: [
          "when do I climb at the south gym?",
          "where do I go bouldering on Tuesday evenings?",
          "do I climb on Tuesdays at the south bouldering gym?",
          "what do I do on Tuesday nights for climbing?",
          "Tuesday evening climbing at the south gym",
          "which day of the week do I visit the south bouldering gym?",
          "I go to the south bouldering gym every Tuesday evening",
          "where do I climb on Tuesday nights?",
          "what time and day do I go to the south bouldering gym?",
          "Tuesday evening is my time at the south bouldering gym",
          "do I climb at the south gym on Tuesday evenings?",
          "what is my Tuesday evening bouldering schedule?",
          "remind me which gym I climb at on Tuesday nights",
          "I do indoor climbing at the south gym on Tuesday evenings",
          "which evening do I go to the south bouldering gym?",
          "bouldering at the south gym on Tuesday evening",
          "where am I on Tuesday evenings for climbing?",
          "am I at the south climbing gym on Tuesday night?",
          "my Tuesday evening routine at the south bouldering gym",
          "tell me about my Tuesday night climbing at the south gym"
        ]
      },
      {
        id: "t2_m2",
        text: "Tuesday nights I go indoor bouldering at the south gym",
        phrasings: [
          "what do I do Tuesday nights at the south gym?",
          "do I go indoor bouldering on Tuesday nights?",
          "indoor bouldering on Tuesday nights at the south gym",
          "what sport do I do on Tuesday nights?",
          "where do I do indoor bouldering on Tuesday nights?",
          "Tuesday night bouldering at the south gym",
          "I do indoor bouldering every Tuesday night at the south gym",
          "is Tuesday night my indoor bouldering night?",
          "tell me what I do on Tuesday night at the south gym",
          "indoor bouldering session on Tuesday nights",
          "do I boulder indoors at the south gym on Tuesday evenings?",
          "what's my Tuesday night athletic activity?",
          "Tuesday nights are for indoor bouldering at south gym",
          "where do I boulder indoors on Tuesdays?",
          "remind me about my Tuesday night indoor bouldering",
          "indoor bouldering on Tuesday nights",
          "am I bouldering indoors at the south gym on Tuesday nights?",
          "my Tuesday night indoor bouldering at the south gym",
          "what happens on Tuesday night at the south gym?",
          "indoor bouldering plans for Tuesday night"
        ]
      },
      {
        id: "t2_m3",
        text: "The south climbing gym is where I work on my bouldering projects",
        phrasings: [
          "where do I work on my bouldering projects?",
          "which climbing gym has my bouldering projects?",
          "I work on bouldering projects at the south climbing gym",
          "where are my current bouldering projects located?",
          "the south climbing gym is my project gym for bouldering",
          "do I project boulders at the south climbing gym?",
          "what gym do I go to for project bouldering?",
          "my bouldering projects are at the south climbing gym",
          "which facility do I use to work on hard boulder problems?",
          "tell me where I work on bouldering projects",
          "south climbing gym bouldering projects",
          "is the south gym where I project boulders?",
          "where do I try to send my bouldering projects?",
          "the south climbing gym bouldering wall projects",
          "where are my bouldering projects?",
          "I climb my bouldering projects at the south gym",
          "which gym do I project at for bouldering?",
          "working on bouldering projects at the south gym",
          "what bouldering gym do I project at?",
          "my bouldering project spot is the south climbing gym"
        ]
      }
    ],
    theme_queries: [
      "what is my bouldering and climbing schedule?",
      "tell me all about my climbing gym and routine",
      "where and when do I climb?",
      "summarize my indoor bouldering habits",
      "what do you know about my climbing at the south gym?",
      "tell me about my Tuesday climbing habits and projects",
      "which climbing gym do I go to and what do I do there?",
      "what are my bouldering days and locations?",
      "what information do you have on my bouldering?",
      "when do I go bouldering?",
      "everything about my climbing routine at the south gym",
      "give me an overview of my bouldering activities",
      "what are my plans for climbing during the week?",
      "how often and where do I boulder?",
      "tell me about my indoor climbing at the south gym",
      "what days do I climb and where are my projects?",
      "my bouldering schedule and gym details",
      "what do I do for rock climbing or bouldering?",
      "where do I train for bouldering?",
      "recap my climbing gym sessions and routine"
    ],
    near_miss: {
      id: "t2_nm",
      text: "I lift free weights at the downtown fitness center on Thursday mornings",
      phrasings: [
        "when do I lift free weights at the downtown fitness center?",
        "where do I go to lift weights on Thursday mornings?",
        "do I lift free weights downtown on Thursdays?",
        "what do I do Thursday morning at the fitness center?",
        "Thursday morning weightlifting downtown",
        "which gym do I go to on Thursday mornings for weights?",
        "I lift weights at the downtown fitness center on Thursday mornings",
        "what is my Thursday morning gym routine?",
        "free weights workout on Thursday morning downtown",
        "tell me about my Thursday morning lifting session",
        "do I workout downtown on Thursday mornings?",
        "lifting weights at the downtown gym on Thursdays",
        "my Thursday morning workout schedule",
        "where do I train with free weights?",
        "Thursday morning gym visit downtown"
      ]
    }
  },
  {
    id: "sister-naima",
    name: "Sister Naima",
    members: [
      {
        id: "t3_m1",
        text: "My sister Naima teaches chemistry at the local high school",
        phrasings: [
          "what subject does my sister Naima teach?",
          "where does my sister Naima work as a teacher?",
          "my sister Naima is a high school chemistry teacher",
          "what does my sister Naima do for work?",
          "does my sister Naima teach chemistry at the local high school?",
          "what school does sister Naima teach at?",
          "my sister Naima teaches high school chemistry",
          "who teaches chemistry at the local high school?",
          "tell me about my sister Naima's teaching job",
          "sister Naima is a chemistry educator at the high school",
          "what is sister Naima's profession?",
          "where does my sister teach chemistry?",
          "my sister Naima, what is her job?",
          "is my sister Naima a chemistry teacher at the high school?",
          "what does Naima my sister teach?",
          "my sister Naima works as a chemistry teacher at the local school",
          "remind me what subject Naima teaches",
          "sister Naima high school chemistry teaching",
          "what grade or subject does my sister Naima teach?",
          "tell me what my sister Naima teaches at the local high school"
        ]
      },
      {
        id: "t3_m2",
        text: "Naima is my sister and she works as a high school chemistry teacher",
        phrasings: [
          "who is Naima and what is her job?",
          "Naima is my sister, right?",
          "what is Naima's relationship to me and what is her career?",
          "is Naima my sister who teaches high school chemistry?",
          "Naima is my sister and a chemistry teacher at high school",
          "what does my sister Naima work as?",
          "confirm that Naima is my sister who teaches high school chemistry",
          "who in my family teaches chemistry?",
          "Naima, my sister, works as a high school chemistry teacher",
          "tell me about Naima's role as my sister and teacher",
          "my sister's name is Naima and she teaches high school chemistry",
          "is Naima my sister?",
          "Naima is my sister, a high school teacher in chemistry",
          "what relation is Naima to me and what does she teach?",
          "Naima works as a high school chemistry teacher and is my sister",
          "does my sister Naima work at a high school teaching chemistry?",
          "who is Naima in my family?",
          "Naima is my sister who teaches high school chemistry",
          "remind me about Naima being my sister and teaching chemistry",
          "how is Naima related to me and what does she do?"
        ]
      },
      {
        id: "t3_m3",
        text: "I call my sister Naima every Sunday evening to catch up",
        phrasings: [
          "when do I call my sister Naima?",
          "do I call sister Naima on Sunday evenings?",
          "what time on Sunday do I talk to my sister Naima?",
          "calling my sister Naima every Sunday evening",
          "who do I call every Sunday evening to catch up?",
          "I catch up with my sister Naima by phone on Sunday evenings",
          "what is my Sunday evening phone call routine with my sister?",
          "how often do I call my sister Naima?",
          "every Sunday evening I call my sister Naima",
          "do I speak with sister Naima on Sunday nights?",
          "remind me to call my sister Naima this Sunday evening",
          "Sunday evening call with sister Naima",
          "when do my sister Naima and I talk on the phone?",
          "I have a weekly Sunday evening call with my sister Naima",
          "catching up with sister Naima on Sunday evening",
          "what evening do I call my sister Naima to chat?",
          "my Sunday evening phone call with sister Naima",
          "do I call Naima every Sunday night?",
          "when is my regular call with my sister Naima?",
          "Sunday evening catch up call with sister Naima"
        ]
      }
    ],
    theme_queries: [
      "tell me everything about my sister Naima",
      "what do you know about my sister Naima, her job, and our calls?",
      "summarize all information about my sister Naima",
      "who is my sister Naima and how do we keep in touch?",
      "what is my relationship and routine with my sister Naima?",
      "tell me about Naima in my life",
      "what does my sister do and when do I speak with her?",
      "give me an overview of my sister Naima",
      "what are the facts about my sister Naima?",
      "everything regarding my sister Naima",
      "how do I stay in touch with my sister and what is her career?",
      "tell me about sister Naima's profession and our weekly schedule",
      "what information is stored about my sister Naima?",
      "what do I have on record about my sister Naima?",
      "recap my sister Naima's job and our Sunday calls",
      "who is Naima and what is our family routine?",
      "tell me about my sister who teaches chemistry",
      "when do I talk to my sister and what does she teach?",
      "details about my sister Naima",
      "all notes on sister Naima"
    ],
    near_miss: {
      id: "t3_nm",
      text: "My coworker Naima is a frontend engineer on the infrastructure team",
      phrasings: [
        "what does coworker Naima do?",
        "who is the Naima on my work team?",
        "my colleague Naima is a frontend engineer in infrastructure",
        "what role does coworker Naima have?",
        "is coworker Naima a frontend engineer?",
        "what team is coworker Naima on at work?",
        "coworker Naima frontend engineering infrastructure team",
        "tell me about my coworker named Naima",
        "who is coworker Naima at my company?",
        "coworker Naima works on frontend infrastructure",
        "what is coworker Naima's job title?",
        "does coworker Naima do frontend engineering on infra?",
        "my coworker Naima's role in the team",
        "colleague Naima frontend engineer",
        "what does my coworker Naima work on?"
      ]
    }
  },
  {
    id: "bookshelf",
    name: "Walnut Bookshelf Project",
    members: [
      {
        id: "t4_m1",
        text: "I am building a custom walnut bookshelf in my garage workshop",
        phrasings: [
          "what am I building in my garage workshop?",
          "what woodworking project am I building in the garage?",
          "I am building a walnut bookshelf in my garage",
          "what custom furniture am I making in the garage workshop?",
          "am I building a walnut bookcase in the garage?",
          "tell me about the custom bookshelf I am building",
          "where am I building my walnut bookshelf?",
          "custom walnut bookshelf build in the garage workshop",
          "what is being built in my garage workshop right now?",
          "building a custom walnut bookshelf in the garage",
          "what kind of wood am I using for the bookshelf project in the garage?",
          "is my garage workshop project a walnut bookshelf?",
          "custom walnut bookcase project in my garage",
          "what am I constructing in my garage workshop?",
          "making a custom walnut bookshelf in the workshop",
          "remind me what I am building out of walnut in the garage",
          "my garage workshop walnut bookshelf build",
          "what furniture piece am I crafting in the garage?",
          "I am making a walnut bookshelf in my garage workshop",
          "tell me about my garage bookshelf project"
        ]
      },
      {
        id: "t4_m2",
        text: "The walnut bookcase in my garage is six feet tall with four shelves",
        phrasings: [
          "how tall is the walnut bookcase in my garage?",
          "how many shelves does my walnut bookcase have?",
          "what are the dimensions of the walnut bookcase in my garage?",
          "six feet tall with four shelves is the walnut bookcase in my garage",
          "how many shelves are on the walnut bookcase in the garage?",
          "is the walnut bookcase in my garage six feet tall?",
          "specs of the walnut bookcase: six feet tall, 4 shelves",
          "the bookcase in my garage has four shelves and is 6 ft high",
          "what are the shelf count and height of the walnut bookcase?",
          "tell me the height and shelf count of the walnut bookcase in the garage",
          "walnut bookcase dimensions in garage: 6 feet tall, four shelves",
          "how big is the walnut bookcase in my garage?",
          "does my walnut bookcase have four shelves?",
          "height of the walnut bookcase in the garage",
          "the walnut bookcase in my garage is six feet high with four shelves",
          "measurements of the walnut bookcase in my garage",
          "how many shelves on the 6 foot walnut bookcase in the garage?",
          "four shelves on the six foot walnut bookcase in the garage",
          "remind me how tall the walnut bookcase in my garage is",
          "what are the specifications of the garage walnut bookcase?"
        ]
      },
      {
        id: "t4_m3",
        text: "I am applying oil finish to my handmade walnut bookcase",
        phrasings: [
          "what finish am I putting on my handmade walnut bookcase?",
          "am I applying oil finish to the walnut bookcase?",
          "oil finish on my handmade walnut bookcase",
          "what am I coating my handmade walnut bookcase with?",
          "I am oil finishing my handmade walnut bookcase",
          "what stage of finishing is the handmade walnut bookcase at?",
          "oil finish application on handmade walnut bookcase",
          "how am I finishing the handmade walnut bookcase?",
          "tell me about applying oil finish to the walnut bookcase",
          "applying oil to the handmade walnut bookcase",
          "what type of finish does my handmade walnut bookcase get?",
          "am I oiling the handmade walnut bookcase?",
          "finishing my handmade walnut bookcase with oil",
          "oil finish for the handmade walnut bookcase",
          "what am I applying to the walnut bookcase right now?",
          "is the handmade walnut bookcase getting an oil finish?",
          "applying an oil finish to my handmade walnut bookcase",
          "coating the handmade walnut bookcase with oil finish",
          "remind me what finish I am applying to the walnut bookcase",
          "oil finishing the walnut bookcase I built"
        ]
      }
    ],
    theme_queries: [
      "tell me everything about my walnut bookshelf project",
      "what are all the details of the bookcase I am building in the garage?",
      "summarize my garage woodworking bookshelf project",
      "what furniture am I making and what are its specs and finish?",
      "all information on my walnut bookcase in the garage",
      "how is my walnut bookcase build progressing?",
      "what do you know about the walnut shelves in my garage workshop?",
      "tell me about the bookshelf I'm building, its dimensions and finish",
      "recap my woodworking project in the garage",
      "give me an overview of the handmade walnut bookcase",
      "everything about my garage workshop furniture build",
      "what are the specs and current status of my walnut bookcase?",
      "what woodworking am I doing in the garage?",
      "describe the custom walnut bookcase project",
      "what do I have recorded about the walnut bookshelf?",
      "dimensions, finish, and details of my walnut bookcase",
      "the garage walnut bookcase project summary",
      "what handmade furniture am I creating in the garage?",
      "status and specs of the walnut bookshelf build",
      "details of my walnut bookcase woodworking"
    ],
    near_miss: {
      id: "t4_nm",
      text: "I bought a flat-pack pine desk from IKEA for my study",
      phrasings: [
        "where did I buy my study desk?",
        "did I buy a flat-pack pine desk from IKEA?",
        "what desk do I have in my study?",
        "IKEA flat-pack pine desk in my study",
        "what kind of desk is in my study?",
        "is my study desk a flat-pack pine desk from IKEA?",
        "tell me about the desk in my study",
        "did I get a pine desk from IKEA for the study?",
        "where is the flat-pack pine desk from IKEA located?",
        "flat-pack pine study desk from IKEA",
        "who made my study desk?",
        "what furniture did I buy from IKEA for my study?",
        "the pine desk in my study from IKEA",
        "remind me where my study desk came from",
        "flat pack pine desk IKEA purchase"
      ]
    }
  },
  {
    id: "allergy",
    name: "Penicillin Allergy",
    members: [
      {
        id: "t5_m1",
        text: "I have a severe allergic reaction to penicillin and amoxicillin",
        phrasings: [
          "what drugs am I severely allergic to?",
          "am I allergic to penicillin and amoxicillin?",
          "severe allergic reaction to penicillin and amoxicillin",
          "what happens if I take penicillin or amoxicillin?",
          "do I have a severe allergy to penicillin antibiotics?",
          "can I take amoxicillin or penicillin?",
          "what antibiotics cause me severe allergic reactions?",
          "I have a severe reaction to amoxicillin and penicillin",
          "remind me of my severe allergy to penicillin and amoxicillin",
          "is amoxicillin safe for me or am I allergic?",
          "penicillin and amoxicillin severe allergy warning",
          "what medications do I have a severe allergic reaction to?",
          "severe reaction to penicillin and amoxicillin drugs",
          "am I allergic to amoxicillin?",
          "never give me penicillin or amoxicillin because of severe allergy",
          "which common antibiotics am I severely allergic to?",
          "allergic reaction to penicillin and amoxicillin",
          "tell me about my severe penicillin allergy",
          "can doctors prescribe me penicillin or amoxicillin?",
          "severe allergic reaction risk: penicillin and amoxicillin"
        ]
      },
      {
        id: "t5_m2",
        text: "My medical chart notes an anaphylactic allergy to penicillin antibiotics",
        phrasings: [
          "what does my medical chart say about penicillin?",
          "is an anaphylactic allergy to penicillin noted in my medical chart?",
          "does my medical chart record an anaphylactic penicillin allergy?",
          "what allergy is listed on my medical chart?",
          "anaphylactic allergy to penicillin antibiotics on medical chart",
          "what kind of reaction to penicillin is in my chart?",
          "my chart notes anaphylactic allergy to penicillin antibiotics",
          "is my penicillin allergy classified as anaphylactic on my chart?",
          "medical chart notes: anaphylaxis to penicillin antibiotics",
          "what critical antibiotic allergy is on my medical chart?",
          "anaphylactic penicillin allergy recorded on chart",
          "does my doctor know about my anaphylactic penicillin allergy?",
          "medical record notes anaphylactic reaction to penicillin",
          "penicillin antibiotics anaphylaxis on my medical chart",
          "what does my official medical record note about allergies?",
          "is anaphylaxis from penicillin antibiotics documented in my chart?",
          "medical chart allergy to penicillin antibiotics",
          "tell me what allergy my medical chart specifies",
          "confirm the allergy noted on my medical chart",
          "anaphylactic allergy to penicillin in my medical chart"
        ]
      },
      {
        id: "t5_m3",
        text: "I wear a medical alert bracelet for my penicillin allergy",
        phrasings: [
          "what does my medical alert bracelet say?",
          "do I wear a medical alert bracelet for penicillin allergy?",
          "why do I wear a medical alert bracelet?",
          "medical alert bracelet for my penicillin allergy",
          "what is engraved or noted on my medical alert bracelet?",
          "I wear a medical alert bracelet for penicillin",
          "does my medical bracelet mention penicillin allergy?",
          "what jewelry do I wear for my penicillin allergy?",
          "my medical alert bracelet notes my penicillin allergy",
          "tell me about my medical alert bracelet",
          "wearing a medical alert bracelet for penicillin allergy",
          "what allergy is on my medical alert bracelet?",
          "do I have a medical alert bracelet?",
          "penicillin allergy medical alert bracelet",
          "why do I have a medical alert bracelet on?",
          "I wear a bracelet alerting doctors to my penicillin allergy",
          "what warning is on my medical alert bracelet?",
          "medical alert bracelet penicillin",
          "remind me why I wear a medical alert bracelet",
          "does my medical alert bracelet warn about penicillin?"
        ]
      }
    ],
    theme_queries: [
      "what are all my allergies and medical warnings?",
      "tell me everything about my penicillin allergy",
      "summarize my medical allergy records and precautions",
      "what antibiotics must I avoid and what precautions do I take?",
      "what is documented about my penicillin and amoxicillin allergies?",
      "give me an overview of my penicillin allergy notes",
      "what medical alert or chart warnings do I have?",
      "everything related to my penicillin allergy",
      "what medications are dangerous for me to take?",
      "can I be given penicillin-class antibiotics?",
      "recap my severe antibiotic allergy and medical bracelet",
      "what health alerts and drug allergies do I have?",
      "all information about my penicillin medical precautions",
      "what should EMTs or doctors know about my drug allergies?",
      "tell me about my allergy bracelet and penicillin reactions",
      "full summary of my penicillin allergy details",
      "what precautions do I take for my drug allergy?",
      "medical chart and bracelet details for penicillin allergy",
      "what are my life-threatening or severe allergies?",
      "drug allergy overview and warnings"
    ],
    near_miss: {
      id: "t5_nm",
      text: "I take daily vitamin D and zinc supplements with breakfast",
      phrasings: [
        "what daily supplements do I take with breakfast?",
        "do I take vitamin D and zinc every morning?",
        "daily vitamin D and zinc with breakfast",
        "what vitamins do I have with breakfast?",
        "I take zinc and vitamin D supplements every day with breakfast",
        "what supplements do I take in the morning with food?",
        "daily supplements: vitamin D and zinc at breakfast",
        "tell me about my morning vitamin D and zinc supplements",
        "do I take zinc with breakfast?",
        "what pills or supplements do I take with my morning meal?",
        "vitamin D and zinc breakfast routine",
        "morning supplements with breakfast",
        "do I take vitamin D daily?",
        "what dietary supplements do I take with breakfast?",
        "daily vitamin D and zinc intake"
      ]
    }
  }
];

const DISTRACTORS_DATA = [
  {
    id: "d1",
    text: "The capital of Oregon is Salem, which was incorporated in 1857",
    phrasings: [
      "what is the capital of Oregon?",
      "when was Salem Oregon incorporated?",
      "is Salem the capital of Oregon?",
      "what year was the Oregon capital incorporated?",
      "Salem is the capital of Oregon, incorporated in 1857",
      "tell me about the capital city of Oregon",
      "in what year was Salem incorporated?",
      "Oregon state capital and incorporation year",
      "what city serves as Oregon's capital?",
      "capital of Oregon incorporation date"
    ]
  },
  {
    id: "d2",
    text: "My car is a blue 2018 Honda Civic with manual transmission",
    phrasings: [
      "what kind of car do I drive?",
      "what color and year is my Honda Civic?",
      "does my Honda Civic have a manual transmission?",
      "blue 2018 Honda Civic manual gearbox",
      "what car do I own and what transmission does it have?",
      "is my 2018 Honda Civic blue?",
      "tell me the specs of my car",
      "my car: 2018 Honda Civic in blue with manual transmission",
      "what transmission is in my blue Civic?",
      "what year is my blue manual Honda Civic?"
    ]
  },
  {
    id: "d3",
    text: "I renewed my passport at the downtown post office last September",
    phrasings: [
      "where did I renew my passport?",
      "did I renew my passport downtown last September?",
      "when did I renew my passport at the post office?",
      "downtown post office passport renewal last September",
      "what did I do at the downtown post office last September?",
      "where was my passport renewed?",
      "last September passport renewal location",
      "did I go to the post office downtown to renew my passport?",
      "passport renewal at downtown post office",
      "when and where did I get my passport renewed?"
    ]
  },
  {
    id: "d4",
    text: "Apollo 11 landed on the lunar surface on July 20, 1969",
    phrasings: [
      "when did Apollo 11 land on the moon?",
      "what date did Apollo 11 touch down on the lunar surface?",
      "Apollo 11 moon landing date July 20 1969",
      "what mission landed on the lunar surface on July 20, 1969?",
      "did Apollo 11 land on the moon in 1969?",
      "exact date of the Apollo 11 lunar landing",
      "July 20 1969 Apollo 11 moon landing",
      "when was the first human lunar landing?",
      "Apollo 11 lunar surface landing date",
      "what happened on July 20, 1969 regarding Apollo 11?"
    ]
  },
  {
    id: "d5",
    text: "I replaced the furnace filter with an allergen-rated pleated filter",
    phrasings: [
      "what kind of filter did I put in the furnace?",
      "did I replace the furnace filter with an allergen-rated one?",
      "allergen-rated pleated filter in the furnace",
      "what furnace filter did I install?",
      "I replaced my furnace filter with a pleated allergen filter",
      "what type of air filter is in the furnace?",
      "furnace filter replacement with allergen-rated pleated filter",
      "did I put an allergen-rated filter in my furnace?",
      "tell me about the furnace filter replacement",
      "what did I replace the furnace filter with?"
    ]
  }
];

const EMBED_CACHE_PATH = path.join(__dirname, "fire-together-embeds.json");
const crypto = require("crypto");

function textHash(s) {
  return crypto.createHash("sha256").update(s).digest("hex").slice(0, 16);
}

async function batchEmbed(texts) {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error("GEMINI_API_KEY not set");
  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents?key=${apiKey}`;

  let embedCache = {};
  if (fs.existsSync(EMBED_CACHE_PATH)) {
    try {
      embedCache = JSON.parse(fs.readFileSync(EMBED_CACHE_PATH, "utf8"));
    } catch {}
  }

  // Find missing texts
  const missing = texts.filter((t) => !embedCache[textHash(t)]);
  console.log(`Embedding cache: ${Object.keys(embedCache).length} cached, ${missing.length} missing.`);

  const BATCH_SIZE = 50;
  for (let i = 0; i < missing.length; i += BATCH_SIZE) {
    const chunk = missing.slice(i, i + BATCH_SIZE);
    process.stdout.write(`Embedding batch ${Math.floor(i / BATCH_SIZE) + 1}/${Math.ceil(missing.length / BATCH_SIZE)} (${chunk.length} items)... `);

    let success = false;
    for (let attempt = 0; attempt < 10; attempt++) {
      try {
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

        if (res.status === 429) {
          const body = await res.json().catch(() => ({}));
          const delaySec = 15;
          process.stdout.write(`[429 rate limit, waiting ${delaySec}s]... `);
          await new Promise((r) => setTimeout(r, delaySec * 1000));
          continue;
        }

        if (!res.ok) {
          throw new Error(`batchEmbedContents error ${res.status}: ${await res.text()}`);
        }

        const data = await res.json();
        for (let j = 0; j < chunk.length; j++) {
          embedCache[textHash(chunk[j])] = data.embeddings[j].values;
        }
        fs.writeFileSync(EMBED_CACHE_PATH, JSON.stringify(embedCache));
        success = true;
        break;
      } catch (err) {
        if (err.message.includes("429")) {
          process.stdout.write(`[429, waiting 15s]... `);
          await new Promise((r) => setTimeout(r, 15000));
          continue;
        }
        throw err;
      }
    }

    if (!success) throw new Error("Failed batch after 10 attempts");
    console.log("done.");
    await new Promise((r) => setTimeout(r, 1000));
  }

  return texts.map((t) => embedCache[textHash(t)]);
}

async function main() {
  console.log("Building 4.0b Hermes Fire-Together Corpus...");

  const allRecords = [];
  const allQueries = [];

  // Assemble records and queries
  for (const theme of THEMES_DATA) {
    for (const m of theme.members) {
      allRecords.push({ id: m.id, text: m.text, theme_id: theme.id, role: "member" });
      for (const p of m.phrasings) {
        allQueries.push({
          query: p,
          kind: "member-paraphrase",
          target_id: m.id,
          theme_id: theme.id,
        });
      }
    }

    // Theme queries
    for (const tq of theme.theme_queries) {
      allQueries.push({
        query: tq,
        kind: "theme-query",
        target_id: null,
        theme_id: theme.id,
      });
    }

    // Near-miss
    allRecords.push({
      id: theme.near_miss.id,
      text: theme.near_miss.text,
      theme_id: theme.id,
      role: "near-miss",
    });
    for (const nmp of theme.near_miss.phrasings) {
      allQueries.push({
        query: nmp,
        kind: "near-miss-query",
        target_id: theme.near_miss.id,
        theme_id: theme.id,
      });
    }
  }

  // Distractors
  for (const d of DISTRACTORS_DATA) {
    allRecords.push({ id: d.id, text: d.text, theme_id: null, role: "distractor" });
    for (const dp of d.phrasings) {
      allQueries.push({
        query: dp,
        kind: "distractor-query",
        target_id: d.id,
        theme_id: null,
      });
    }
  }

  console.log(`Planted layout:`);
  console.log(`  Facts total: ${allRecords.length} (15 on-theme, 5 near-miss, 5 distractors)`);
  console.log(`  Queries total: ${allQueries.length} (300 member, 100 theme, 75 near-miss, 50 distractors)`);

  const recordTexts = allRecords.map((r) => r.text);
  const queryTexts = allQueries.map((q) => q.query);
  const allTexts = [...recordTexts, ...queryTexts];

  console.log(`Embedding ${allTexts.length} total strings with gemini-embedding-001 (768 dim)...`);
  const embeddings = await batchEmbed(allTexts);

  for (let i = 0; i < allRecords.length; i++) {
    allRecords[i].embedding = embeddings[i];
  }
  for (let i = 0; i < allQueries.length; i++) {
    allQueries[i].embedding = embeddings[allRecords.length + i];
  }

  // True theme pairs: all combinations of members in the same theme
  const true_pairs = [];
  for (const theme of THEMES_DATA) {
    const ids = theme.members.map((m) => m.id);
    for (let i = 0; i < ids.length; i++) {
      for (let j = i + 1; j < ids.length; j++) {
        true_pairs.push({ a: ids[i], b: ids[j], theme_id: theme.id });
      }
    }
  }

  // Near miss pairs: each member paired with the theme near-miss
  const near_miss_pairs = [];
  for (const theme of THEMES_DATA) {
    const nmId = theme.near_miss.id;
    for (const m of theme.members) {
      near_miss_pairs.push({ a: m.id, b: nmId, theme_id: theme.id });
    }
  }

  // Unrelated pairs: across different themes or with distractors (excluding near-miss pairs)
  const unrelated_pairs = [];
  const allIds = allRecords.map((r) => r.id);
  const themeMap = new Map(allRecords.map((r) => [r.id, r.theme_id]));

  for (let i = 0; i < allIds.length; i++) {
    for (let j = i + 1; j < allIds.length; j++) {
      const a = allIds[i], b = allIds[j];
      const tA = themeMap.get(a), tB = themeMap.get(b);
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
    themes: THEMES_DATA.map((t) => ({ id: t.id, name: t.name })),
    records: allRecords,
    queries: allQueries,
    true_pairs,
    near_miss_pairs,
    unrelated_pairs,
  };

  fs.mkdirSync(path.dirname(CORPUS_PATH), { recursive: true });
  fs.writeFileSync(CORPUS_PATH, JSON.stringify(corpus, null, 2), "utf8");
  console.log(`Corpus written successfully to ${CORPUS_PATH}!`);
  console.log(`  File size: ${(fs.statSync(CORPUS_PATH).size / (1024 * 1024)).toFixed(2)} MB`);
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

module.exports = { main, THEMES_DATA, DISTRACTORS_DATA };
