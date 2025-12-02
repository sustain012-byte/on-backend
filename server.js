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
// Vertex AI TTS — Leda 음성 생성 (수정 버전)
// ======================================================
//
// lines: ["문장1","문장2",...]
// → base64 WAV 배열 반환
//
async function synthesizeLinesWithVertexTTS(lines = []) {
  if (!Array.isArray(lines) || !lines.length) return [];

  // Vertex 설정이 없으면 그냥 전부 null 반환
  if (!VERTEX_API_KEY || !VERTEX_PROJECT_ID) {
    console.warn("⚠️ Vertex 설정이 없어 TTS를 건너뜁니다.");
    return lines.map(() => null);
  }

  const results = [];

  for (const text of lines) {
    if (!text) {
      results.push(null);
      continue;
    }

    // ✅ Vertex AI Gemini 2.5 Flash TTS 요청 포맷
    const body = {
      contents: [
        {
          role: "user",
          parts: [{ text }]
        }
      ],
      tools: [
        {
          speech_synthesis: {
            voice: {
              // AI Studio 데모에서 사용한 한국어 Leda
              voice_name: "Leda",
              language_code: "ko-KR"
            }
          }
        }
      ],
      generation_config: {
        // 오디오 형식 지정
        response_mime_type: "audio/wav"
      }
    };

    const url =
      `https://${VERTEX_LOCATION}-aiplatform.googleapis.com/v1/` +
      `projects/${VERTEX_PROJECT_ID}/locations/${VERTEX_LOCATION}` +
      `/publishers/google/models/gemini-2.5-flash-tts:generateContent`;

    try {
      const res = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          // ✅ key는 헤더로
          "x-goog-api-key": VERTEX_API_KEY
        },
        body: JSON.stringify(body)
      });

      if (!res.ok) {
        const errTxt = await res.text().catch(() => "");
        console.error("[Vertex TTS HTTP ERROR]", res.status, errTxt);
        results.push(null);
        continue;
      }

      const data = await res.json();

      // ✅ 응답 구조에서 base64 오디오 꺼내기
      // candidates[0].content.parts[*].inline_data.data
      const parts = data?.candidates?.[0]?.content?.parts || [];
      const inlinePart = parts.find(
        (p) => p.inline_data && p.inline_data.data
      );
      const base64audio = inlinePart?.inline_data?.data || null;

      if (!base64audio) {
        console.warn("[Vertex TTS] 오디오 데이터를 찾지 못했습니다.", JSON.stringify(data).slice(0, 200) + "...");
      }

      results.push(base64audio);

    } catch (e) {
      console.error("[Vertex TTS EXCEPTION]: ", e);
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
너는 ACT(수용전념치료) 기반의 한국어 상담 코치이다.
사용자가 작성한 일기를 읽고, 그 안의 경험을 ACT 관점의 4개 범주로 분류한다.

[4개 범주 정의]
1) situation: 사건, 맥락, 환경, 타인과의 상호작용 등 "무엇이 일어났는가".
2) feeling: 감정(정서) + 신체감각. 단순 감정 단어가 아닌 구체적 경험.
3) thought: 해석, 평가, 자동적 사고, 자기비판, 미래 예상, 의미부여.
4) behavior: 실제 행동, 선택, 말, 몸의 반응. 심리적 회피·접근은 단어로 쓰지 말고 자연스러운 행동 문장으로 기술.

[출력 규칙]
- 각 범주마다 정확히 3문장.
- 모든 문장은 25자 이내, '~다.'로 끝나는 평서문.
- 입력에 없는 내용을 상상하거나 꾸며내지 않는다.
- 감정과 생각, 생각과 행동을 혼합하지 않는다.
- behavior에는 "접근/회피/수용" 같은 단어를 절대 넣지 않는다.
- JSON 이외의 텍스트는 절대 출력하지 않는다.

[출력 형식]
{
  "situation": { "cards": [ {"text":""}, {"text":""}, {"text":""} ] },
  "feeling":   { "cards": [ {"text":""}, {"text":""}, {"text":""} ] },
  "thought":   { "cards": [ {"text":""}, {"text":""}, {"text":""} ] },
  "behavior":  { "cards": [ {"text":""}, {"text":""}, {"text":""} ] }
}

반드시 위 JSON 형식만 반환하라.
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
- 원문 사실을 그대로 사용하고 새로 꾸미지 않는다.
- ACT 개념(탈융합, 수용, 현재 머물기, 가치, 전념행동)을 자연스럽게 녹인다.
- 명령형, 질문형, 조언형 금지.
- 모든 문장은 따뜻한 자기진술문, '~다.'로 끝난다.
- 1문장 30~40자 이내, 총 7문장.
- JSON 형식으로만 출력.
- JSON 외 텍스트 절대 금지.

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
