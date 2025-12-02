// ======================================================
// server.js — Vertex AI TTS (Leda) 버전 완성본
// ======================================================

const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');

const app = express();

// ================== 환경변수 ==================
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const VERTEX_API_KEY = process.env.VERTEX_API_KEY;        // NEW ★
const VERTEX_PROJECT_ID = process.env.VERTEX_PROJECT_ID;  // NEW ★
const VERTEX_LOCATION = "asia-northeast3";                // 한국 리전

if (!OPENAI_API_KEY) console.warn("⚠️ OPENAI_API_KEY 없음");
if (!VERTEX_API_KEY) console.warn("⚠️ VERTEX_API_KEY 없음");
if (!VERTEX_PROJECT_ID) console.warn("⚠️ VERTEX_PROJECT_ID 없음");


// ======================================================
// CORS 설정
// ======================================================
app.use(cors({
  origin: "*",
  methods: ["GET","POST","OPTIONS"],
  allowedHeaders: ["Content-Type"],
}));
app.options('*', cors());

app.use(express.json({ limit: '1mb' }));

// 디버깅 로그
app.use((req,res,next)=>{
  console.log(`[REQ] ${req.method} ${req.path}`);
  next();
});


// ======================================================
// OpenAI 호출 유틸
// ======================================================
async function callOpenAI(model, temperature, systemMsg, userJson) {
  const payload = {
    model,
    messages: [
      { role: 'system', content: systemMsg },
      { role: 'user',   content: JSON.stringify(userJson) }
    ],
    response_format: { type: 'json_object' }
  };

  // gpt-5 계열은 temperature 안 넣음
  if (!/^gpt-5/.test(model) && typeof temperature === "number") {
    payload.temperature = temperature;
  }

  const res = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${OPENAI_API_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error("OPENAI ERROR " + err);
  }

  const data = await res.json();

  let raw = data?.choices?.[0]?.message?.content ?? "{}";
  raw = raw.replace(/^```json/,"").replace(/```$/,"").trim();

  return JSON.parse(raw);
}


// ======================================================
// 텍스트 정리
// ======================================================
function normalizeDa(t){
  let s = String(t||"").trim();
  s = s.replace(/["']/g,"")
       .replace(/[\uD800-\uDBFF][\uDC00-\uDFFF]/g,"")
       .replace(/[?!…]+$/,"")
       .trim();
  return s;
}


// ======================================================
// Vertex AI TTS — Leda 음성 생성
// ======================================================
//
// lines: ["문장1","문장2",...]
// → base64 WAV 배열 반환
//
async function synthesizeLinesWithVertexTTS(lines = []) {
  if (!Array.isArray(lines) || !lines.length) return [];

  const results = [];

  for (const text of lines) {
    if (!text) {
      results.push(null);
      continue;
    }

    const body = {
      contents: [
        {
          role: "user",
          parts: [{ text }]
        }
      ],
      generation_config: {
        response_mime_type: "audio/wav",
        voice_name: "Leda"   // ★ 바로 여기! Leda 화자
      }
    };

    const url =
      `https://${VERTEX_LOCATION}-aiplatform.googleapis.com/v1/projects/${VERTEX_PROJECT_ID}/locations/${VERTEX_LOCATION}/publishers/google/models/gemini-2.5-flash-tts:generateContent?key=${VERTEX_API_KEY}`;

    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });

      const data = await res.json();

      const base64audio =
        data?.candidates?.[0]?.content?.parts?.[0]?.audio?.data || null;

      results.push(base64audio);

    } catch (e) {
      console.error("[Vertex TTS ERROR]: ", e);
      results.push(null);
    }
  }

  return results;
}


// ======================================================
// 프롬프트들
// ======================================================
const PROMPTS = {
  classifySuggest: {
    system: `
너는 ACT(수용전념치료) 관점에서 한국어 일기를 읽고,
경험을 네 가지 범주로 정리해 주는 상담사이다.

네 가지 범주는 다음과 같다:
- situation
- feeling
- thought
- behavior

규칙:
- 각 범주마다 3문장씩 만든다.
- 모든 문장은 25자 이내이며 반드시 '~다.'로 끝난다.
- 입력에 없는 내용을 상상하여 쓰지 않는다.
- 출력은 반드시 JSON 객체 하나로만 한다.
- JSON 외의 다른 텍스트는 절대 포함되지 않아야 한다.

형식(JSON):
{
  "situation": { "cards": [ { "text": "" }, { "text": "" }, { "text": "" } ] },
  "feeling":   { "cards": [ { "text": "" }, { "text": "" }, { "text": "" } ] },
  "thought":   { "cards": [ { "text": "" }, { "text": "" }, { "text": "" } ] },
  "behavior":  { "cards": [ { "text": "" }, { "text": "" }, { "text": "" } ] }
}
    `.trim()
  },

  practice: {
    system: `
너는 ACT(수용전념치료) 기반의 한국어 심리 코치이다.
사용자의 일기 내용을 읽고, 그 안의 감정·생각·행동을 자연스럽게 재해석하여
짧고 따뜻한 문장 7개를 만든다.

목표:
- 사용자가 자신의 경험을 새롭게 바라보고, 수용과 전념의 시각으로 이해하게 돕는다.
- 각 문장은 그날의 구체적 경험에 밀착하면서도 자기이해를 촉진해야 한다.

규칙:
- 원문 사실(사건, 감정, 생각, 행동)을 그대로 사용하고 새로 꾸미지 않는다.
- ACT 개념(탈융합, 수용, 현재 머물기, 가치, 전념행동)을 자연스럽게 녹인다.
- 명령형, 질문형, 조언형, “~해야 한다”는 표현 금지.
- 모든 문장은 따뜻한 자기진술문으로, ‘~다.’로 끝난다.
- 1문장 30~40자 이내, 총 7문장.
- 반드시 JSON 형식으로만 출력한다.
- JSON 외 텍스트는 절대 포함하지 않는다.

형식(JSON):
{
  "practice_sets_json": [
    {"text": "문장1"},
    {"text": "문장2"},
    ...
    {"text": "문장7"}
  ]
}
    `.trim()
  }
};



// ======================================================
// /classifysuggest
// ======================================================
app.post("/classifysuggest", async (req,res)=>{
  try{
    let { text="" } = req.body;
    text = text.slice(0,3000);

    const out = await callOpenAI(
      "gpt-4.1-mini",
      null,
      PROMPTS.classifySuggest.system,
      { text }
    );

    const TOP_K=3;
    function clean(arr){
      return (arr||[])
        .slice(0,TOP_K)
        .map(c=>({text:normalizeDa(c.text||"")}))
        .filter(c=>c.text);
    }

    res.json({
      ok:true,
      used_model:"gpt-4.1-mini",
      result:{
        situation:{cards:clean(out?.situation?.cards)},
        feeling:{cards:clean(out?.feeling?.cards)},
        thought:{cards:clean(out?.thought?.cards)},
        behavior:{cards:clean(out?.behavior?.cards)}
      }
    });

  }catch(e){
    console.error(e);
    res.status(500).json({ok:false,error:e.message});
  }
});


// ======================================================
// /practice
// ======================================================
app.post("/practice", async (req,res)=>{
  try{
    let { text="" } = req.body;
    text = text.slice(0,3000);

    const out = await callOpenAI(
      "gpt-5.1",
      0.2,
      PROMPTS.practice.system,
      { text }
    );

    let arr = [];

    if (Array.isArray(out.practice_sets_json)) {
      arr = out.practice_sets_json;
    } else if (Array.isArray(out.sentences)) {
      arr = out.sentences.map(s=>({text:s.text||s}));
    }

    arr = arr.slice(0,7)
             .map(x=>({text:normalizeDa(x.text)}))
             .filter(Boolean);

    while(arr.length<7){
      arr.push({text:"나는 지금의 나를 있는 그대로 둔다"});
    }

    const lines = arr.map(x=>x.text);

    // ★ Vertex Leda TTS 호출
    const audioList = await synthesizeLinesWithVertexTTS(lines);

    res.json({
      ok:true,
      used_model:"gpt-5.1",
      practice_sets_json:arr,
      audio_base64_list:audioList,
      tts:{
        provider:"vertex-ai",
        voice:"Leda",
        model:"gemini-2.5-flash-tts"
      }
    });

  }catch(e){
    console.error(e);
    res.status(500).json({ok:false,error:e.message});
  }
});


// ======================================================
app.get("/", (_,res)=>res.send("ON backend is running (Vertex TTS Leda)"));


// ======================================================
const PORT = process.env.PORT || 3000;
app.listen(PORT, ()=>{
  console.log(`🚀 ON backend running on ${PORT}`);
});
