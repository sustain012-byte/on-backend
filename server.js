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

  // gpt-5 계열은 temperature 고정이라면 건드리지 않음
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

// ======== 프롬프트 (더 단순화 / 2문장 고정 / text만) ========

const PROMPTS = {
  classifySuggest: {
    system: `
너는 한국어 일기를 ACT(수용전념치료) 관점으로 4영역으로만 나누어 제안하는 도우미다.
영역은 situation, feeling, thought, behavior 네 가지다.

규칙:
- 각 영역마다 짧은 한글 문장 "정확히 2개"를 만든다.
- 입력에 없는 사실은 새로 만들지 않는다.
- 문장은 25자 이내의 평서문으로, '~다.'로 끝낸다.
- feeling은 지금 느끼는 감정, thought는 해석/평가, situation은 사건/상황, behavior는 실제 행동을 쓴다.
- behavior 문장 안에는 '접근', '수용', '회피'라는 단어를 쓰지 말고, 그냥 '~했다/하지 않았다.' 형태의 행동만 자연스럽게 쓴다.
- confidence, tags 같은 값은 만들지 말고, 각 카드에는 text만 포함한다.
- 아래 JSON 형식을 정확히 지키고, 그 외의 말은 하지 않는다.

반환(JSON 하나):
{
  "situation": { "cards": [ { "text": "" }, { "text": "" } ] },
  "feeling":   { "cards": [ { "text": "" }, { "text": "" } ] },
  "thought":   { "cards": [ { "text": "" }, { "text": "" } ] },
  "behavior":  { "cards": [ { "text": "" }, { "text": "" } ] }
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
    let { text = '', lang = 'ko' } = req.body || {};
    text = String(text || '').slice(0, 3000);

    // 각 영역당 2문장 고정
    const TOP_K = 2;

    if (!text) {
      return res.status(400).json({ ok:false, error:'empty_text' });
    }

    const out = await callOpenAI(
      'gpt-4.1-turbo',              // 🔹 여기서 nano → 4.1-turbo
      0.2,
      PROMPTS.classifySuggest.system,
      { text, lang, top_k: TOP_K }
    );

    function clean(arr) {
      return (Array.isArray(arr) ? arr : [])
        .slice(0, TOP_K)
        .map(c => ({
          // 🔹 text만 남기고 나머지는 버림
          text: normalizeDa(c && c.text || '')
        }))
        .filter(c => c.text);
    }

    const result = {
      situation: { cards: clean(out?.situation?.cards) },
      feeling:   { cards: clean(out?.feeling?.cards) },
      thought:   { cards: clean(out?.thought?.cards) },
      behavior:  { cards: clean(out?.behavior?.cards) }
    };

    return res.json({ ok:true, result, used_model:'gpt-4.1-turbo' });

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
      'gpt-5',           // 🔹 여기 practice는 그대로 gpt-5 유지 (원하면 나중에 4.1-turbo로도 바꿀 수 있음)
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
