// server.js
// Render용 간단 백엔드: /classifysuggest, /practice 두 개 라우트

const express = require('express');
const cors = require('cors');

const app = express();

// 🔐 반드시 Render 대시보드에 OPENAI_API_KEY 환경변수 넣어줘야 함
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

if (!OPENAI_API_KEY) {
  console.warn('⚠️ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.');
}

// 공통 미들웨어
app.use(cors());                  // 개발 편하게 전체 origin 허용
app.use(express.json({ limit: '1mb' }));

// ======== OpenAI 호출 유틸 ========

async function callOpenAI(model, temperature, systemMsg, userJson) {
  if (!OPENAI_API_KEY) {
    throw new Error('missing_openai_key');
  }

  const payload = {
    model,
    messages: [
      { role: 'system', content: systemMsg },
      { role: 'user',   content: JSON.stringify(userJson) }
    ],
    response_format: { type: 'json_object' }
  };

  if (!/^gpt-5(?:-|$)/.test(model) && typeof temperature === 'number') {
    payload.temperature = temperature;
  }

  const t0 = Date.now();
  const res = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${OPENAI_API_KEY}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(payload)
  });
  const t1 = Date.now();
  console.log(`[OPENAI] model=${model} elapsed=${t1 - t0}ms`);

  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`openai_http_${res.status}: ${text}`);
  }

  const data = await res.json();
  let txt =
    data?.choices?.[0]?.message?.content ??
    '{}';

  // ```json ... ``` 감싸져 오는 경우 제거
  txt = String(txt).replace(/^```json/, '').replace(/```$/, '').trim();

  return JSON.parse(txt || '{}');
}

// ======== 텍스트 정리 유틸 (GAS 버전과 동일하게) ========

function normalizeDa(t) {
  let s = String(t || '').trim();
  s = s
    .replace(/["']/g, '')
    .replace(/[\uD800-\uDBFF][\uDC00-\uDFFF]/g, '') // 이모지 제거
    .replace(/[?!…]+$/,'')
    .trim();
  if (!s) return '';
  return s;
}

// ======== 프롬프트 (GAS에서 쓰던 것 그대로) ========

const PROMPTS = {
  classifySuggest: {
    system: `
입력된 일기 텍스트를 ACT(수용전념치료) 관점으로 4영역으로 제안한다.
각 영역별 2~3개의 짧은 문장을 제안한다.

규칙:
- 같은 의미나 감정의 중복 문장은 제거한다.
- 감정은 현재의 느낌을, 생각은 해석/평가를, 행동은 회피·수용·접근 중 하나로 표현한다.
- 불분명하면 "구름이가 이 부분은 도와줄 수 없어요."로 남긴다.

반환(JSON 하나):
{
  "situation": { "cards": [ { "text": "" }, ... ] },
  "feeling":   { "cards": [ { "text": "" }, ... ] },
  "thought":   { "cards": [ { "text": "" }, ... ] },
  "behavior":  { "cards": [ { "text": "" }, ... ] }
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
- 고유명사는 OO으로 치환.
- JSON 하나만 출력.

형식:
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

// ======== 라우트: /classifysuggest ========

app.post('/classifysuggest', async (req, res) => {
  try {
    let { text = '', lang = 'ko', top_k = 3 } = req.body || {};
    text = String(text || '').slice(0, 3000);
    top_k = Math.max(1, Math.min(3, parseInt(top_k || 2, 10)));

    if (!text) {
      return res.status(400).json({ ok:false, error:'empty_text' });
    }

    const out = await callOpenAI(
      'gpt-5-nano',
      0.2,
      PROMPTS.classifySuggest.system,
      { text, lang, top_k }
    );

    function clean(arr) {
      return (Array.isArray(arr) ? arr : [])
        .slice(0, top_k)
        .map(c => ({
          text: normalizeDa(c && c.text || ''),
          confidence: Math.max(
            0.5,
            Math.min(0.95, Number(c && c.confidence || 0.62))
          )
        }))
        .filter(c => c.text);
    }

    const result = {
      situation: { cards: clean(out?.situation?.cards) },
      feeling:   { cards: clean(out?.feeling?.cards) },
      thought:   { cards: clean(out?.thought?.cards) },
      behavior:  { cards: clean(out?.behavior?.cards) }
    };

    return res.json({ ok:true, result, used_model:'gpt-5-nano' });

  } catch (err) {
    console.error('[/classifysuggest] error', err);
    return res.status(500).json({
      ok:false,
      error: err.message || 'server_error'
    });
  }
});

// ======== 라우트: /practice ========

app.post('/practice', async (req, res) => {
  try {
    let { text = '', lang = 'ko' } = req.body || {};
    text = String(text || '').slice(0, 3000);

    if (!text) {
      return res.status(400).json({ ok:false, error:'empty_text' });
    }

    const out = await callOpenAI(
      'gpt-5',
      0.2,
      PROMPTS.practice.system,
      { text, lang }
    );

    let arr = [];
    if (out && Array.isArray(out.practice_sets_json)) {
      arr = out.practice_sets_json;
    } else if (out && Array.isArray(out.sentences)) {
      arr = out.sentences.map(s => ({ text: s && s.text ? s.text : s }));
    }

    arr = (arr || [])
      .slice(0, 7)
      .map(item => {
        const t = normalizeDa(item && item.text || '');
        return t ? { text: t } : null;
      })
      .filter(Boolean);

    // 7개 안 채워지면 기본 문장으로 채우기
    while (arr.length < 7) {
      arr.push({ text: normalizeDa('나는 지금의 나를 있는 그대로 둔다') });
    }

    return res.json({ ok:true, practice_sets_json: arr, used_model:'gpt-5' });

  } catch (err) {
    console.error('[/practice] error', err);
    return res.status(500).json({
      ok:false,
      error: err.message || 'server_error'
    });
  }
});

// ======== 헬스 체크 ========

app.get('/', (req, res) => {
  res.send('ON backend is running');
});

// ======== 서버 시작 ========

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`✅ ON backend listening on port ${PORT}`);
});
