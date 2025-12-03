// ======================================================
// server.js — OpenAI + Google Gemini TTS(Leda) 버전
// ======================================================

const express = require('express');
const cors = require('cors');

const app = express();

// ================== 환경변수 ==================
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

// Gemini 키는 이름이 헷갈릴 수 있어서 후보 몇 개를 다 확인한다
const GEMINI_API_KEY =
  process.env.GEMINI_API_KEY ||   // 우리가 지금 쓰려는 이름
  process.env.Gemini_API ||       // 혹시 이렇게 만들어 둔 경우
  process.env.gemini_api ||       // 전부 소문자로 만든 경우
  '';                             // 아무 것도 없으면 빈 문자열

// 디버깅: 실제로 process.env에 어떤 'gemini' 관련 키가 있는지 출력
const geminiEnvKeys = Object.keys(process.env).filter(k =>
  k.toLowerCase().includes('gemini')
);
console.log('[ENV] GEMINI 관련 키들:', geminiEnvKeys);

if (!OPENAI_API_KEY) {
  console.warn('⚠️ OPENAI_API_KEY 없음');
}
if (!GEMINI_API_KEY) {
  console.warn('⚠️ GEMINI_API_KEY 없음 — TTS는 텍스트만 동작');
} else {
  console.log('✅ GEMINI_API_KEY 감지:', GEMINI_API_KEY.slice(0, 8) + '...');
}

// ======================================================
// CORS 설정
// ======================================================
app.use(cors({
  origin: '*',
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type'],
}));
app.options('*', cors());

app.use(express.json({ limit: '1mb' }));

// 디버깅 로그
app.use((req, res, next) => {
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
      { role: 'user', content: JSON.stringify(userJson) },
    ],
    response_format: { type: 'json_object' },
  };

  // gpt-5 계열은 temperature 안 넣기
  if (!/^gpt-5/.test(model) && typeof temperature === 'number') {
    payload.temperature = temperature;
  }

  const t0 = Date.now();
  const res = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${OPENAI_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });
  const elapsed = Date.now() - t0;
  console.log(`[OPENAI] model=${model} elapsed=${elapsed}ms status=${res.status}`);

  if (!res.ok) {
    const err = await res.text().catch(() => '');
    throw new Error('OPENAI ERROR ' + err);
  }

  const data = await res.json();

  let raw = data?.choices?.[0]?.message?.content ?? '{}';
  raw = raw.replace(/^```json/, '').replace(/```$/, '').trim();

  return JSON.parse(raw || '{}');
}

// ======================================================
// 텍스트 정리
// ======================================================
function normalizeDa(t) {
  let s = String(t || '').trim();
  s = s
    .replace(/["']/g, '')
    .replace(/[\uD800-\uDBFF][\uDC00-\uDFFF]/g, '') // 이모지 제거
    .replace(/[?!…]+$/, '')
    .trim();
  return s;
}

// ======================================================
// Google Gemini TTS — Leda 음성 생성
// ======================================================
//
// lines: ["문장1", "문장2", ...]
// → ["base64오디오1", "base64오디오2", ...]
//
async function synthesizeLinesWithGeminiTTS(lines = []) {
  if (!Array.isArray(lines) || !lines.length) return [];

  if (!GEMINI_API_KEY) {
    console.warn('⚠️ GEMINI_API_KEY 없음 — TTS 건너뜀');
    return lines.map(() => null);
  }

  const MODEL_ID = "gemini-2.5-flash-preview-tts";
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${MODEL_ID}:generateContent?key=${encodeURIComponent(
    GEMINI_API_KEY,
  )}`;

  // ⭐ 각 문장(text)마다 비동기 작업(Promise)을 하나씩 만든다
  const tasks = lines.map((text) => {
    if (!text) {
      // 비어 있으면 그냥 null 리턴하는 Promise
      return Promise.resolve(null);
    }

    const body = {
      contents: [
        {
          role: 'user',
          parts: [{ text }],
        },
      ],
      generationConfig: {
        responseModalities: ['audio'],  // 소문자 audio
        temperature: 1,
        speech_config: {
          voice_config: {
            prebuilt_voice_config: {
              voice_name: 'Leda',
            },
          },
        },
      },
    };

    // 🔹 여기부터는 "한 문장에 대한 TTS 요청"을 비동기 처리
    return (async () => {
      try {
        const t0 = Date.now();
        const res = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        const elapsed = Date.now() - t0;
        console.log(
          `[GEMINI_TTS] len=${text.length} elapsed=${elapsed}ms status=${res.status}`,
        );

        if (!res.ok) {
          const errTxt = await res.text().catch(() => '');
          console.error('[GEMINI_TTS HTTP ERROR]', res.status, errTxt);
          return null;
        }

        const data = await res.json();

        let payload = data;
        if (Array.isArray(payload)) {
          payload =
            payload.find(ch => ch?.candidates?.[0]?.content?.parts?.length) ||
            payload[0];
        }

        const parts = payload?.candidates?.[0]?.content?.parts || [];
        let base64audio = null;

        for (const p of parts) {
          if (p.inlineData && p.inlineData.data) {
            base64audio = p.inlineData.data;
            break;
          }
          if (p.inline_data && p.inline_data.data) {
            base64audio = p.inline_data.data;
            break;
          }
          if (p.audio && p.audio.data) {
            base64audio = p.audio.data;
            break;
          }
        }

        if (!base64audio) {
          console.warn(
            '[GEMINI_TTS] 오디오 데이터를 찾지 못했습니다.',
            JSON.stringify(payload).slice(0, 200) + '...',
          );
        }

        return base64audio || null;
      } catch (e) {
        console.error('[GEMINI_TTS EXCEPTION]', e);
        return null;
      }
    })();
  });

  // ⭐ 병렬 실행: 모든 문장 TTS 요청을 동시에 보낸 뒤, 한 번에 결과 수집
  const results = await Promise.all(tasks);
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
    `.trim(),
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
    `.trim(),
  },
};

// ======================================================
// /classifysuggest
// ======================================================
app.post('/classifysuggest', async (req, res) => {
  try {
    let { text = '' } = req.body || {};
    text = text.slice(0, 3000);

    const out = await callOpenAI(
      'gpt-4.1-mini',
      null,
      PROMPTS.classifySuggest.system,
      { text },
    );

    const TOP_K = 3;
    function clean(arr) {
      return (arr || [])
        .slice(0, TOP_K)
        .map((c) => ({ text: normalizeDa(c.text || '') }))
        .filter((c) => c.text);
    }

    res.json({
      ok: true,
      used_model: 'gpt-4.1-mini',
      result: {
        situation: { cards: clean(out?.situation?.cards) },
        feeling: { cards: clean(out?.feeling?.cards) },
        thought: { cards: clean(out?.thought?.cards) },
        behavior: { cards: clean(out?.behavior?.cards) },
      },
    });
  } catch (e) {
    console.error('[/classifysuggest] error', e);
    res.status(500).json({ ok: false, error: e.message });
  }
});

// ======================================================
// /practice
// ======================================================
app.post('/practice', async (req, res) => {
  try {
    let { text = '' } = req.body || {};
    text = text.slice(0, 3000);

    const out = await callOpenAI(
      'gpt-5.1',
      0.2,
      PROMPTS.practice.system,
      { text },
    );

    let arr = [];

    if (Array.isArray(out.practice_sets_json)) {
      arr = out.practice_sets_json;
    } else if (Array.isArray(out.sentences)) {
      arr = out.sentences.map((s) => ({ text: s.text || s }));
    }

    arr = arr
      .slice(0, 7)
      .map((x) => ({ text: normalizeDa(x.text) }))
      .filter(Boolean);

    while (arr.length < 7) {
      arr.push({ text: '나는 지금의 나를 있는 그대로 둔다' });
    }

    const lines = arr.map((x) => x.text);

    // Gemini Leda TTS 호출
    const audioList = await synthesizeLinesWithGeminiTTS(lines);

// --- /practice 응답 부분 ---
res.json({
  ok: true,
  used_model: 'gpt-5.1',
  practice_sets_json: arr,
  audio_base64_list: audioList,
  tts: {
    provider: 'google-gemini',
    voice: 'Leda',
    model: 'gemini-2.5-flash-preview-tts',
  },
});
  } catch (e) {
    console.error('[/practice] error', e);
    res.status(500).json({ ok: false, error: e.message });
  }
});

// ======================================================
app.get('/', (_, res) =>
  res.send('ON backend is running (Gemini Leda TTS)'),
);

// ======================================================
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`🚀 ON backend running on ${PORT}`);
});
