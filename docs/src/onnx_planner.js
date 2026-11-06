/**
 * Browser & Node ONNX Runtime Planner for `gnumanth/xkcd-artist` (`model_quantized.onnx`, 1.72 MB Int8).
 *
 * Implements exact token-ID parity with `training/generate_dataset.py:encode_prompt_tokens`
 * and executes all 7 neural heads (`layout_logits`, `prop_logits`, `poses`, `bubbles`,
 * `style_vec`, `ink_patch`, `invalid_logit`) via WebAssembly SIMD (`onnxruntime-web`).
 */

export const HF_MODEL_REPO = 'https://huggingface.co/gnumanth/xkcd-artist';
export const HF_ONNX_URL = 'https://huggingface.co/gnumanth/xkcd-artist/resolve/main/onnx/model_quantized.onnx';
const ORT_CDN_ESM = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.wasm.min.mjs';
const ORT_WASM_PREFIX = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/';

export const PLANNER_CHARACTERS = [
  'cueball',
  'megan',
  'black_hat',
  'beret_guy',
  'ponytail',
  'hairbun',
];

export const PLANNER_LAYOUTS = [
  'single_panel',
  'two_panel',
  'three_panel',
  'four_panel',
  'annotated_chart',
  'flowchart_diagram',
];

export const PLANNER_PROPS = [
  'none',
  'laptop_desk',
  'whiteboard',
  'server_rack',
  'pointer_stick',
  'couch',
  'chart_board',
  'pedestal',
];

// Exact 160-entry VOCAB_WORDS list from `training/generate_dataset.py`
const VOCAB_WORDS = [
  '<pad>', '<unk>', 'sudo', 'make', 'me', 'a', 'sandwich', 'regex', 'git', 'commit',
  'push', 'merge', 'conflict', 'compiler', 'code', 'compiling', 'sword', 'fighting',
  'server', 'cloud', 'kubernetes', 'docker', 'dependency', 'nebraska', 'tower', 'stack',
  'ai', 'machine', 'learning', 'linear', 'algebra', 'pile', 'stir', 'until', 'answers',
  'python', 'import', 'antigravity', 'flying', 'dynamic', 'typing', 'pointers', 'memory',
  'leak', 'segfault', 'null', 'timezone', 'utc', 'iso', '8601', 'date', 'format',
  'standard', 'standards', '14', '15', 'competing', 'universal', 'password', 'entropy',
  'correct', 'horse', 'battery', 'staple', 'security', 'crypto', 'rsa', 'wrench',
  'quantum', 'physics', 'math', 'integral', 'proof', 'statistic', 'correlation',
  'causation', 'p_value', 'significant', 'jelly', 'beans', 'acne', 'green', 'chart',
  'graph', 'axis', 'exponential', 'extrapolating', 'husband', 'yesterday', 'zero',
  'dojo', 'emacs', 'vim', 'nano', 'butterfly', 'cosmic', 'ray', 'flip', 'bit',
  'cueball', 'megan', 'black', 'hat', 'beret', 'guy', 'ponytail', 'hairbun',
  'desk', 'laptop', 'whiteboard', 'pointer', 'couch', 'server', 'rack', 'chart',
  'someone', 'is', 'wrong', 'on', 'the', 'internet', 'duty', 'calls', 'sleep',
  '3am', 'debugging', 'production', 'database', 'sql', 'injection', 'bobby', 'tables',
  'drop', 'table', 'students', 'sanitize', 'inputs', 'xkcd', 'comic', 'panel',
  'dialogue', 'speech', 'bubble', 'stick', 'figure', 'monochrome', 'ink', 'drawing',
];

// Note: Python `{w: i for i, w in enumerate(VOCAB_WORDS)}` overwrites duplicate keys ('server', 'chart') with later index
const WORD_TO_ID = new Map();
VOCAB_WORDS.forEach((w, idx) => {
  WORD_TO_ID.set(w, idx);
});

/**
 * Compact RFC 1321 MD5 returning the first 8 hex digits as an unsigned 32-bit integer
 * (`int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16)`).
 */
const MD5_K = Int32Array.from({ length: 64 }, (_, i) => Math.floor(Math.abs(Math.sin(i + 1)) * 4294967296) | 0);
const MD5_S = [
  7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
  5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
  4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
  6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];

export function md5First8HexUint32(str) {
  const bytes = new TextEncoder().encode(str);
  const bitLen = bytes.length * 8;
  const padLen = ((56 - ((bytes.length + 1) % 64)) + 64) % 64;
  const buf = new Uint8Array(bytes.length + 1 + padLen + 8);
  buf.set(bytes, 0);
  buf[bytes.length] = 0x80;
  const view = new DataView(buf.buffer);
  view.setUint32(buf.length - 8, bitLen >>> 0, true);
  view.setUint32(buf.length - 4, Math.floor(bitLen / 4294967296) >>> 0, true);

  let a0 = 0x67452301 | 0;
  let b0 = 0xefcdab89 | 0;
  let c0 = 0x98badcfe | 0;
  let d0 = 0x10325476 | 0;

  for (let offset = 0; offset < buf.length; offset += 64) {
    let A = a0;
    let B = b0;
    let C = c0;
    let D = d0;
    for (let i = 0; i < 64; i++) {
      let F = 0;
      let g = 0;
      if (i < 16) {
        F = (B & C) | (~B & D);
        g = i;
      } else if (i < 32) {
        F = (D & B) | (~D & C);
        g = (5 * i + 1) % 16;
      } else if (i < 48) {
        F = B ^ C ^ D;
        g = (3 * i + 5) % 16;
      } else {
        F = C ^ (B | ~D);
        g = (7 * i) % 16;
      }
      const word = view.getInt32(offset + g * 4, true);
      const sum = (A + F + MD5_K[i] + word) | 0;
      const s = MD5_S[i];
      const rot = ((sum << s) | (sum >>> (32 - s))) | 0;
      A = D;
      D = C;
      C = B;
      B = (B + rot) | 0;
    }
    a0 = (a0 + A) | 0;
    b0 = (b0 + B) | 0;
    c0 = (c0 + C) | 0;
    d0 = (d0 + D) | 0;
  }

  // First 4 little-endian bytes of MD5 digest formatted as 8 hex chars -> big-endian uint32
  const out = new Uint8Array(4);
  new DataView(out.buffer).setInt32(0, a0, true);
  return ((out[0] << 24) | (out[1] << 16) | (out[2] << 8) | out[3]) >>> 0;
}

export function encodePromptTokens(text, maxLen = 36) {
  const matches = String(text).toLowerCase().match(/[a-z0-9_]+/g) || [];
  const ids = [];
  const baseOffset = VOCAB_WORDS.length;
  const bucketSpan = 1200 - baseOffset;

  for (const tok of matches.slice(0, maxLen)) {
    if (WORD_TO_ID.has(tok)) {
      ids.push(WORD_TO_ID.get(tok));
    } else {
      const h = md5First8HexUint32(tok);
      ids.push(baseOffset + (h % bucketSpan));
    }
  }
  while (ids.length < maxLen) {
    ids.push(0);
  }
  return ids;
}

function softmax(arr) {
  let maxVal = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] > maxVal) maxVal = arr[i];
  }
  let sum = 0;
  const exps = new Float64Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    const e = Math.exp(arr[i] - maxVal);
    exps[i] = e;
    sum += e;
  }
  return Array.from(exps, v => Number((v / (sum || 1)).toFixed(4)));
}

let _ortModulePromise = null;
let _sessionPromise = null;
let _loadedModelSource = null;

export async function loadOnnxPlanner({
  modelUrls = ['./model_quantized.onnx', HF_ONNX_URL],
  ortModule = null,
} = {}) {
  if (_sessionPromise) return _sessionPromise;

  _sessionPromise = (async () => {
    const ort = ortModule || (await (_ortModulePromise ||= import(/* @vite-ignore */ ORT_CDN_ESM)));
    if (ort.env?.wasm) {
      ort.env.wasm.wasmPaths = ORT_WASM_PREFIX;
      ort.env.wasm.numThreads = 1;
    }

    let lastErr = null;
    for (const url of modelUrls) {
      try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status} fetching ${url}`);
        const buf = await res.arrayBuffer();
        const session = await ort.InferenceSession.create(buf, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all',
        });
        _loadedModelSource = url.includes('huggingface.co') ? 'huggingface:gnumanth/xkcd-artist' : 'local:model_quantized.onnx';
        return { ort, session, source: _loadedModelSource };
      } catch (err) {
        lastErr = err;
      }
    }
    _sessionPromise = null;
    throw lastErr || new Error('Unable to load model_quantized.onnx');
  })();

  return _sessionPromise;
}

/**
 * Execute the 5-head `XkcdArtistForGeneration` + LoRA ink patch ONNX graph on a comic spec.
 */
export async function runOnnxPlanner(enrichedPrompt, cast = ['cueball', 'megan'], options = {}) {
  const { ort, session, source } = await loadOnnxPlanner(options);
  const t0 = performance.now();

  const tokIds = encodePromptTokens(enrichedPrompt, 36);
  const paddedCast = [...cast, 'cueball', 'megan'].slice(0, 2);
  const castIdx = paddedCast.map(c => {
    const idx = PLANNER_CHARACTERS.indexOf(c);
    return idx >= 0 ? idx : 0;
  });

  const tokenTensor = new ort.Tensor('int64', BigInt64Array.from(tokIds, v => BigInt(v)), [1, 36]);
  const castTensor = new ort.Tensor('int64', BigInt64Array.from(castIdx, v => BigInt(v)), [1, 2]);

  const outs = await session.run({
    token_ids: tokenTensor,
    cast_ids: castTensor,
  });

  const latencyMs = Number((performance.now() - t0).toFixed(2));
  const layoutLogits = Array.from(outs.layout_logits.data);
  const propLogits = Array.from(outs.prop_logits.data);
  const styleVec = Array.from(outs.style_vec.data);
  const poses = Array.from(outs.poses.data);
  const bubbles = Array.from(outs.bubbles.data);
  const inkPatch = Array.from(outs.ink_patch.data);
  const invLogit = Number(outs.invalid_logit.data[0]);

  const layoutProbs = softmax(layoutLogits);
  const propProbs = softmax(propLogits);
  const predLayoutIdx = layoutProbs.indexOf(Math.max(...layoutProbs));
  const predPropIdx = propProbs.indexOf(Math.max(...propProbs));
  const pInvalid = Number((1.0 / (1.0 + Math.exp(-invLogit))).toFixed(4));

  return {
    source,
    model_id: 'gnumanth/xkcd-artist',
    latency_ms: latencyMs,
    predicted_layout: PLANNER_LAYOUTS[predLayoutIdx],
    layout_confidence: layoutProbs[predLayoutIdx],
    predicted_prop: PLANNER_PROPS[predPropIdx],
    prop_confidence: propProbs[predPropIdx],
    style: {
      wobble_amp_px: Number(Math.max(1.1, Math.min(2.4, styleVec[0])).toFixed(4)),
      stroke_width_px: Number(Math.max(1.5, Math.min(2.8, styleVec[1])).toFixed(4)),
      bubble_zone_ratio: Number(Math.max(0.18, Math.min(0.45, styleVec[2])).toFixed(4)),
      ink_contrast: Number(Math.max(0.75, Math.min(1.0, styleVec[3])).toFixed(4)),
    },
    p_invalid: pInvalid,
    poses_12dof: poses.map(v => Number(v.toFixed(4))),
    bubble_boxes: bubbles.map(v => Number(v.toFixed(4))),
    ink_patch_mean: Number((inkPatch.reduce((a, b) => a + b, 0) / inkPatch.length).toFixed(4)),
  };
}
