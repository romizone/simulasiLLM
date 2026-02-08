const SENTENCES = [
  "Data visualization helps teams understand sales trends faster.",
  "Large language models learn patterns from billions of text tokens.",
  "The operations team prepares a daily report before the meeting starts.",
  "Customers ask for solutions that are clear, fast, and measurable.",
  "The security system detects anomalies and sends instant notifications.",
  "Market analysis shows rising demand for sustainable products.",
  "Software engineers optimize queries to improve response efficiency.",
  "The company designs a new strategy for regional expansion this year.",
  "Automated testing reduces critical bugs before application release.",
  "The project lead ensures every risk is monitored regularly.",
];

const MODEL = {
  layers: 12,
  heads: 4,
  hiddenSize: 16,
  headDim: 4,
};

const sentenceSelect = document.getElementById("sentenceSelect");
const headSelect = document.getElementById("headSelect");
const tempRange = document.getElementById("tempRange");
const topkRange = document.getElementById("topkRange");
const genCountRange = document.getElementById("genCountRange");
const tempValue = document.getElementById("tempValue");
const topkValue = document.getElementById("topkValue");
const genCountValue = document.getElementById("genCountValue");
const headCountLabel = document.getElementById("headCountLabel");
const generateBtn = document.getElementById("generateBtn");
const resetBtn = document.getElementById("resetBtn");

const tokenStream = document.getElementById("tokenStream");
const attentionMatrix = document.getElementById("attentionMatrix");
const attentionBars = document.getElementById("attentionBars");
const qkvSnapshot = document.getElementById("qkvSnapshot");
const nextTokenBox = document.getElementById("nextTokenBox");
const generatedTokensEl = document.getElementById("generatedTokens");
const generatedTextEl = document.getElementById("generatedText");

const state = {
  selectedSentence: 0,
  activeHead: 0,
  focusIndex: 0,
  temperature: 0.8,
  topk: 5,
  generateCount: 6,
  baseTokens: [],
  generatedTokens: [],
  parsed: null,
  vocab: [],
  bigrams: {},
  isGenerating: false,
};

function hashString(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i += 1) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function seededRandom(seed) {
  let s = seed >>> 0;
  return function rand() {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function clamp(n, min, max) {
  return Math.max(min, Math.min(max, n));
}

function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i += 1) {
    total += a[i] * b[i];
  }
  return total;
}

function softmax(values) {
  const maxVal = Math.max(...values);
  const exps = values.map((v) => Math.exp(v - maxVal));
  const sum = exps.reduce((acc, v) => acc + v, 0);
  return exps.map((e) => e / (sum || 1));
}

function tokenize(text) {
  const matched = text.match(/\p{L}+[\p{L}\p{N}'-]*|\p{N}+|[.,!?;:]/gu);
  return matched && matched.length ? matched : text.split(/\s+/).filter(Boolean);
}

function normalizeToken(token) {
  return token.toLowerCase();
}

function isPunctuation(token) {
  return /^[.,!?;:]$/.test(token);
}

function detokenize(tokens) {
  if (!tokens.length) return "";
  return tokens.reduce((acc, token, idx) => {
    if (idx === 0) return token;
    if (isPunctuation(token)) return `${acc}${token}`;
    return `${acc} ${token}`;
  }, "");
}

function makeMatrix(rows, cols, seed) {
  const rand = seededRandom(seed);
  const out = [];
  for (let r = 0; r < rows; r += 1) {
    const row = [];
    for (let c = 0; c < cols; c += 1) {
      row.push((rand() * 2 - 1) * 0.9);
    }
    out.push(row);
  }
  return out;
}

function matVec(m, v) {
  return m.map((row) => dot(row, v));
}

function buildHiddenVector(token, position, sentenceSeed) {
  const rand = seededRandom(hashString(`${token}|${position}|${sentenceSeed}`));
  const v = [];
  for (let d = 0; d < MODEL.hiddenSize; d += 1) {
    const semantic = (rand() * 2 - 1) * 0.85;
    const positional = Math.sin((position + 1) / Math.pow(10000, (2 * (d % 2)) / MODEL.hiddenSize) + d * 0.12);
    v.push(semantic + positional * 0.35);
  }
  return v;
}

function buildHeadData(hiddenStates, sentenceSeed, headIndex) {
  const offset = headIndex * MODEL.headDim;
  const Wq = makeMatrix(MODEL.headDim, MODEL.headDim, hashString(`WQ:${sentenceSeed}:${headIndex}`));
  const Wk = makeMatrix(MODEL.headDim, MODEL.headDim, hashString(`WK:${sentenceSeed}:${headIndex}`));
  const Wv = makeMatrix(MODEL.headDim, MODEL.headDim, hashString(`WV:${sentenceSeed}:${headIndex}`));

  const queries = [];
  const keys = [];
  const values = [];

  for (let i = 0; i < hiddenStates.length; i += 1) {
    const slice = hiddenStates[i].slice(offset, offset + MODEL.headDim);
    queries.push(matVec(Wq, slice).map((n) => Math.tanh(n)));
    keys.push(matVec(Wk, slice).map((n) => Math.tanh(n)));
    values.push(matVec(Wv, slice).map((n) => Math.tanh(n)));
  }

  const scores = [];
  const weights = [];

  for (let i = 0; i < hiddenStates.length; i += 1) {
    const rowScores = [];
    for (let j = 0; j < hiddenStates.length; j += 1) {
      if (j > i) {
        rowScores.push(-1e9);
      } else {
        rowScores.push(dot(queries[i], keys[j]) / Math.sqrt(MODEL.headDim));
      }
    }
    scores.push(rowScores);
    weights.push(softmax(rowScores));
  }

  return {
    queries,
    keys,
    values,
    scores,
    weights,
  };
}

function currentSequenceTokens() {
  return [...state.baseTokens, ...state.generatedTokens];
}

function rebuildParsed() {
  const tokens = currentSequenceTokens();
  const sentenceSeed = hashString(SENTENCES[state.selectedSentence]);
  const hiddenStates = tokens.map((tok, idx) => buildHiddenVector(tok, idx, sentenceSeed));
  const heads = [];

  for (let h = 0; h < MODEL.heads; h += 1) {
    heads.push(buildHeadData(hiddenStates, sentenceSeed, h));
  }

  state.parsed = {
    text: detokenize(tokens),
    tokens,
    hiddenStates,
    heads,
  };

  state.focusIndex = tokens.length
    ? clamp(state.focusIndex, 0, tokens.length - 1)
    : 0;
}

function parseCurrentSentence() {
  const text = SENTENCES[state.selectedSentence];
  state.baseTokens = tokenize(text);
  state.generatedTokens = [];
  state.focusIndex = clamp(state.baseTokens.length - 1, 0, state.baseTokens.length - 1);
  rebuildParsed();
}

function initBigramsAndVocab() {
  const bag = new Set();
  const bigrams = {};

  const fallbackWords = [
    "and",
    "for",
    "with",
    "that",
    "the",
    "to",
    "in",
    "is",
    "will",
    "more",
    "fast",
    "efficient",
    "strategy",
    "team",
    "model",
    "system",
    ".",
    ",",
  ];

  SENTENCES.forEach((sentence) => {
    const toks = tokenize(sentence).map(normalizeToken);
    toks.forEach((t) => bag.add(t));

    for (let i = 0; i < toks.length - 1; i += 1) {
      const a = toks[i];
      const b = toks[i + 1];
      if (!bigrams[a]) bigrams[a] = {};
      bigrams[a][b] = (bigrams[a][b] || 0) + 1;
    }
  });

  fallbackWords.forEach((w) => bag.add(w));
  state.vocab = Array.from(bag).sort((a, b) => a.localeCompare(b, "en"));
  state.bigrams = bigrams;
}

function wordEmbedding(word) {
  const rand = seededRandom(hashString(`VOCAB:${word}`));
  const out = [];
  for (let i = 0; i < MODEL.hiddenSize; i += 1) {
    out.push((rand() * 2 - 1) * 0.8);
  }
  return out;
}

function contextFromHead(head, focusIndex) {
  const weights = head.weights[focusIndex];
  const context = new Array(MODEL.headDim).fill(0);

  for (let j = 0; j <= focusIndex; j += 1) {
    const w = weights[j];
    const v = head.values[j];
    for (let d = 0; d < MODEL.headDim; d += 1) {
      context[d] += w * v[d];
    }
  }

  return context;
}

function buildPrediction() {
  const { tokens, heads } = state.parsed;
  if (!tokens.length) return [];

  const focus = clamp(state.focusIndex, 0, tokens.length - 1);
  const currentToken = normalizeToken(tokens[focus]);

  const fullContext = [];
  heads.forEach((head) => {
    fullContext.push(...contextFromHead(head, focus));
  });

  const logits = state.vocab.map((word) => {
    const emb = wordEmbedding(word);
    let logit = dot(fullContext, emb) / Math.sqrt(MODEL.hiddenSize);

    const bigramBoost = state.bigrams[currentToken]?.[word] || 0;
    logit += bigramBoost * 1.5;

    if (word === currentToken) {
      logit -= 0.35;
    }

    return { word, logit };
  });

  logits.sort((a, b) => b.logit - a.logit);

  const k = clamp(state.topk, 1, Math.min(12, logits.length));
  const filtered = logits.slice(0, k);
  const scaled = filtered.map((x) => x.logit / state.temperature);
  const probs = softmax(scaled);

  return filtered.map((item, idx) => ({
    word: item.word,
    prob: probs[idx],
    logit: item.logit,
  }));
}

function sampleFromPredictions(predictions) {
  if (!predictions.length) return null;
  const r = Math.random();
  let acc = 0;

  for (let i = 0; i < predictions.length; i += 1) {
    acc += predictions[i].prob;
    if (r <= acc) return predictions[i].word;
  }

  return predictions[predictions.length - 1].word;
}

function setGenerating(isGenerating) {
  state.isGenerating = isGenerating;
  generateBtn.disabled = isGenerating;
  resetBtn.disabled = isGenerating;
  generateBtn.textContent = isGenerating ? "Generating..." : "Generate";
}

function runGeneration() {
  if (state.isGenerating || !state.parsed) return;

  setGenerating(true);

  const steps = clamp(state.generateCount, 1, 12);
  for (let i = 0; i < steps; i += 1) {
    rebuildParsed();
    state.focusIndex = state.parsed.tokens.length - 1;

    const predictions = buildPrediction();
    const sampled = sampleFromPredictions(predictions);
    if (!sampled) break;

    state.generatedTokens.push(sampled);
  }

  rebuildParsed();
  state.focusIndex = state.parsed.tokens.length - 1;
  renderAll();
  setGenerating(false);
}

function resetGeneration() {
  state.generatedTokens = [];
  rebuildParsed();
  state.focusIndex = state.parsed.tokens.length - 1;
  renderAll();
}

function renderTokenStream() {
  tokenStream.innerHTML = "";
  const generatedStart = state.baseTokens.length;

  state.parsed.tokens.forEach((token, idx) => {
    const btn = document.createElement("button");
    btn.type = "button";

    const classes = ["token"];
    if (idx === state.focusIndex) classes.push("active");
    if (idx >= generatedStart) classes.push("generated");
    btn.className = classes.join(" ");

    btn.innerHTML = `${token}<small>${idx}</small>`;
    btn.title = `Token ${idx}: ${token}`;

    btn.addEventListener("click", () => {
      state.focusIndex = idx;
      renderAll();
    });

    tokenStream.appendChild(btn);
  });
}

function renderAttentionMatrix() {
  const { tokens, heads } = state.parsed;
  const head = heads[state.activeHead];
  const n = tokens.length;

  attentionMatrix.innerHTML = "";

  const grid = document.createElement("div");
  grid.className = "matrix";
  grid.style.gridTemplateColumns = `repeat(${n + 1}, 34px)`;

  const corner = document.createElement("div");
  corner.className = "header-cell corner";
  corner.textContent = "Q\\K";
  grid.appendChild(corner);

  for (let j = 0; j < n; j += 1) {
    const h = document.createElement("div");
    h.className = "header-cell";
    h.textContent = j;
    h.title = `Key ${j}: ${tokens[j]}`;
    grid.appendChild(h);
  }

  for (let i = 0; i < n; i += 1) {
    const rowHead = document.createElement("div");
    rowHead.className = "header-cell row-head";
    rowHead.textContent = i;
    rowHead.title = `Query ${i}: ${tokens[i]}`;
    grid.appendChild(rowHead);

    for (let j = 0; j < n; j += 1) {
      const cell = document.createElement("div");
      const masked = j > i;
      cell.className = `cell ${masked ? "masked" : ""} ${i === state.focusIndex ? "focus-row" : ""}`;

      if (masked) {
        cell.textContent = "×";
      } else {
        const w = head.weights[i][j];
        const alpha = 0.08 + w * 0.85;
        cell.style.background = `rgba(10, 160, 165, ${alpha.toFixed(3)})`;
        cell.textContent = `${Math.round(w * 99)}`;
        cell.title = `Q${i} -> K${j}: ${w.toFixed(4)}`;
      }

      grid.appendChild(cell);
    }
  }

  attentionMatrix.appendChild(grid);
}

function renderAttentionBars() {
  attentionBars.innerHTML = "";
  const { tokens, heads } = state.parsed;
  const head = heads[state.activeHead];
  const row = head.weights[state.focusIndex];

  for (let j = 0; j <= state.focusIndex; j += 1) {
    const wrapper = document.createElement("div");
    wrapper.className = "bar-row";

    const label = document.createElement("div");
    label.className = "bar-label";
    label.textContent = `${tokens[j]} [${j}]`;

    const track = document.createElement("div");
    track.className = "bar-track";

    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = `${(row[j] * 100).toFixed(2)}%`;

    const val = document.createElement("div");
    val.className = "bar-val";
    val.textContent = row[j].toFixed(3);

    track.appendChild(fill);
    wrapper.appendChild(label);
    wrapper.appendChild(track);
    wrapper.appendChild(val);
    attentionBars.appendChild(wrapper);
  }
}

function renderVectorRows(target, vector) {
  vector.forEach((val, idx) => {
    const row = document.createElement("div");
    row.className = "vector-row";

    const dim = document.createElement("div");
    dim.textContent = `d${idx}`;

    const track = document.createElement("div");
    track.className = "vector-track";

    const fill = document.createElement("div");
    fill.className = "vector-fill";
    fill.style.width = `${Math.min(Math.abs(val), 1) * 100}%`;
    fill.style.background =
      val >= 0
        ? "linear-gradient(90deg, #f19848, #ffb173)"
        : "linear-gradient(90deg, #8f6adf, #b59af4)";

    const v = document.createElement("div");
    v.className = "bar-val";
    v.textContent = val.toFixed(3);

    track.appendChild(fill);
    row.appendChild(dim);
    row.appendChild(track);
    row.appendChild(v);
    target.appendChild(row);
  });
}

function renderQKV() {
  qkvSnapshot.innerHTML = "";
  const head = state.parsed.heads[state.activeHead];
  const idx = state.focusIndex;

  const cards = [
    { title: "Query (Q)", data: head.queries[idx] },
    { title: "Key (K)", data: head.keys[idx] },
    { title: "Value (V)", data: head.values[idx] },
  ];

  cards.forEach((card) => {
    const box = document.createElement("div");
    box.className = "vector-box";

    const h3 = document.createElement("h3");
    h3.textContent = card.title;
    box.appendChild(h3);

    renderVectorRows(box, card.data);
    qkvSnapshot.appendChild(box);
  });
}

function renderNextToken() {
  nextTokenBox.innerHTML = "";

  const predictions = buildPrediction();
  const queryToken = state.parsed.tokens[state.focusIndex];
  const topWord = predictions[0]?.word || "-";

  const title = document.createElement("div");
  title.className = "next-title";
  title.textContent = `Query token: "${queryToken}" -> Next token prediction: "${topWord}"`;
  nextTokenBox.appendChild(title);

  predictions.forEach((pred) => {
    const row = document.createElement("div");
    row.className = "next-row";

    const word = document.createElement("div");
    word.className = "next-word";
    word.textContent = pred.word;

    const track = document.createElement("div");
    track.className = "next-track";

    const fill = document.createElement("div");
    fill.className = "next-fill";
    fill.style.width = `${(pred.prob * 100).toFixed(2)}%`;

    const val = document.createElement("div");
    val.className = "next-val";
    val.textContent = `${(pred.prob * 100).toFixed(2)}%`;

    track.appendChild(fill);
    row.appendChild(word);
    row.appendChild(track);
    row.appendChild(val);
    nextTokenBox.appendChild(row);
  });
}

function renderGeneratedOutput() {
  generatedTokensEl.innerHTML = "";

  if (!state.generatedTokens.length) {
    const placeholder = document.createElement("div");
    placeholder.className = "chip";
    placeholder.textContent = "No generated tokens yet.";
    generatedTokensEl.appendChild(placeholder);

    generatedTextEl.textContent =
      "Click Generate to append new tokens. The attention matrix will expand with the sequence length.";
    return;
  }

  state.generatedTokens.forEach((token, idx) => {
    const node = document.createElement("span");
    node.className = "token generated";
    node.innerHTML = `${token}<small>g${idx + 1}</small>`;
    generatedTokensEl.appendChild(node);
  });

  const fullSequence = detokenize(currentSequenceTokens());
  const tokenLabel = state.generatedTokens.length === 1 ? "token" : "tokens";
  generatedTextEl.textContent = `Output (${state.generatedTokens.length} ${tokenLabel}): ${fullSequence}`;
}

function renderAll() {
  renderTokenStream();
  renderAttentionMatrix();
  renderAttentionBars();
  renderQKV();
  renderNextToken();
  renderGeneratedOutput();
}

function initControls() {
  SENTENCES.forEach((sentence, idx) => {
    const opt = document.createElement("option");
    opt.value = String(idx);
    opt.textContent = sentence;
    sentenceSelect.appendChild(opt);
  });

  for (let h = 0; h < MODEL.heads; h += 1) {
    const opt = document.createElement("option");
    opt.value = String(h);
    opt.textContent = `Head ${h + 1}`;
    headSelect.appendChild(opt);
  }

  headCountLabel.textContent = `${MODEL.heads}`;

  sentenceSelect.addEventListener("change", (e) => {
    state.selectedSentence = Number(e.target.value);
    parseCurrentSentence();
    renderAll();
  });

  headSelect.addEventListener("change", (e) => {
    state.activeHead = Number(e.target.value);
    renderAll();
  });

  tempRange.addEventListener("input", (e) => {
    state.temperature = Number(e.target.value);
    tempValue.textContent = state.temperature.toFixed(1);
    renderNextToken();
  });

  topkRange.addEventListener("input", (e) => {
    state.topk = Number(e.target.value);
    topkValue.textContent = String(state.topk);
    renderNextToken();
  });

  genCountRange.addEventListener("input", (e) => {
    state.generateCount = Number(e.target.value);
    genCountValue.textContent = String(state.generateCount);
  });

  generateBtn.addEventListener("click", runGeneration);
  resetBtn.addEventListener("click", resetGeneration);
}

function init() {
  initBigramsAndVocab();
  initControls();
  parseCurrentSentence();

  sentenceSelect.value = String(state.selectedSentence);
  headSelect.value = String(state.activeHead);
  tempValue.textContent = state.temperature.toFixed(1);
  topkValue.textContent = String(state.topk);
  genCountValue.textContent = String(state.generateCount);

  renderAll();
}

init();
