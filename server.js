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

네 가지 범주는 다음과 같다.
- situation
- feeling
- thought
- behavior

각 범주마다 3문장, 25자 이내, "~다."로 끝나는 문장만 출력하라.
입력에 없는 내용 상상 금지.
JSON 외 말 금지.
    `.trim()
  },

  practice: {
    system: `
너는 ACT 기반 한국어 심리 코치다.
사용자가 쓴 일기 내용을 기반으로 따뜻한 자기진술문 7개를 만든다.

규칙:
- 원문 기반, 상상 금지
- ACT(수용·전념) 요소 자연스럽게 녹이기
- "~다."로 끝나는 문장
- 각 문장 30~40자
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
