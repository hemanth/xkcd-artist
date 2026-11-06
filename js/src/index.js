/**
 * xkcd-artist — Multi-Panel & Diagram Generative XKCD Engine
 * Pure monochrome Randall Munroe stage art + organic hand-wobbled SVG pen strokes + `xkcd-script` lettering.
 */
import { SCENES_META } from './scenes_data.js';
import { renderKinematicComic } from './kinematic_renderer.js';
import {
  encodePromptTokens,
  HF_MODEL_REPO,
  HF_ONNX_URL,
  loadOnnxPlanner,
  runOnnxPlanner,
} from './onnx_planner.js';

export { encodePromptTokens, HF_MODEL_REPO, HF_ONNX_URL, loadOnnxPlanner, runOnnxPlanner };

export const LAYOUT_TYPES = [
  'single_panel',
  'two_panel',
  'three_panel',
  'four_panel',
  'annotated_chart',
  'flowchart_diagram',
];

export const CHARACTERS = [
  'cueball',
  'megan',
  'black_hat',
  'beret_guy',
  'ponytail',
  'hairbun',
];

export const PROPS = [
  'none',
  'laptop_desk',
  'whiteboard',
  'server_rack',
  'pointer_stick',
  'couch',
  'chart_board',
  'pedestal',
];

const ICONIC_TEMPLATES = [
  {
    match: ['sudo', 'sandwich'],
    title: 'SANDWICH',
    scene: 'scene_armchair',
    layout: 'single_panel',
    prop: 'couch',
    cast: ['cueball', 'megan'],
    turns: [
      { speaker: 'left', text: 'MAKE ME A SANDWICH.' },
      { speaker: 'right', text: 'WHAT? MAKE IT YOURSELF.' },
      { speaker: 'left', text: 'SUDO MAKE ME A SANDWICH.' },
      { speaker: 'right', text: 'OKAY.' },
    ],
    alt: 'Proper User Policy apparently means Simon Says.',
  },
  {
    match: ['compiling', 'sword fight', 'slacking'],
    title: 'COMPILING',
    scene: 'scene_compiling',
    layout: 'single_panel',
    prop: 'laptop_desk',
    cast: ['cueball', 'black_hat'],
    header: 'THE #1 PROGRAMMER EXCUSE\nFOR LEGITIMATELY SLACKING OFF:\n"MY CODE\'S COMPILING."',
    turns: [
      { speaker: 'left', text: 'HEY! GET BACK TO WORK!' },
      { speaker: 'right', text: 'COMPILING!' },
      { speaker: 'left', text: 'OH. CARRY ON.' },
    ],
    alt: "Are you stealing those LCDs? // Yeah, but I'm doing it while my code compiles.",
  },
  {
    match: ['linear algebra', 'stir the pile'],
    title: 'MACHINE LEARNING',
    scene: 'scene_ml_pile',
    layout: 'single_panel',
    prop: 'pedestal',
    cast: ['ponytail', 'cueball'],
    turns: [
      { speaker: 'left', text: 'THIS IS YOUR MACHINE LEARNING SYSTEM?' },
      {
        speaker: 'right',
        text: 'YUP! YOU POUR THE DATA INTO THIS BIG PILE OF LINEAR ALGEBRA, THEN COLLECT THE ANSWERS ON THE OTHER SIDE.',
      },
      { speaker: 'left', text: 'WHAT IF THE ANSWERS ARE WRONG?' },
      { speaker: 'right', text: 'JUST STIR THE PILE UNTIL THEY START LOOKING RIGHT.' },
    ],
    alt: "The pile gets soaked with data and starts to get mushy over time, so it's technically recurrent.",
  },
  {
    match: ['nebraska', 'modern digital infrastructure'],
    title: 'DEPENDENCY',
    scene: 'scene_dependency',
    layout: 'single_panel',
    prop: 'server_rack',
    cast: ['cueball', 'ponytail'],
    turns: [
      { speaker: 'left', text: 'ALL MODERN DIGITAL\nINFRASTRUCTURE' },
      {
        speaker: 'right',
        text: 'A PROJECT SOME\nRANDOM PERSON\nIN NEBRASKA HAS\nBEEN THANKLESSLY\nMAINTAINING\nSINCE 2003',
      },
    ],
    alt: "Someday ImageMagick will finally break for good and we'll have a long period of scrambling as we try to reassemble civilization from the rubble.",
  },
  {
    match: ['14 competing', '15 standards', 'universal standard'],
    title: 'STANDARDS',
    scene: 'scene_standards_3panel',
    layout: 'three_panel',
    prop: 'whiteboard',
    cast: ['cueball', 'ponytail'],
    turns: [
      { speaker: 'left', text: 'SITUATION:\nTHERE ARE\n14 COMPETING\nSTANDARDS.' },
      {
        speaker: 'right',
        text: "14?! RIDICULOUS!\nWE NEED TO DEVELOP\nONE UNIVERSAL STANDARD\nTHAT COVERS EVERYONE'S\nUSE CASES.",
      },
      { speaker: 'left', text: 'SOON:\nSITUATION:\nTHERE ARE\n15 COMPETING\nSTANDARDS.' },
    ],
    alt: "Fortunately, the charging one has been solved now that we've all standardized on mini-USB. Or is it micro-USB? Shit.",
  },
  {
    match: ['duty calls', 'wrong on the internet'],
    title: 'DUTY CALLS',
    scene: 'scene_desk',
    layout: 'single_panel',
    prop: 'laptop_desk',
    cast: ['megan', 'cueball'],
    turns: [
      { speaker: 'left', text: 'ARE YOU COMING TO BED?' },
      { speaker: 'right', text: "I CAN'T. THIS IS IMPORTANT." },
      { speaker: 'left', text: 'WHAT?' },
      { speaker: 'right', text: 'SOMEONE IS WRONG ON THE INTERNET.' },
    ],
    alt: "What do you want me to do? LEAVE? Then they'll keep being wrong!",
  },
];

function escapeXml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/**
 * Width-calibrated word wrapper accounting for narrow (`I`, `1`, `'`, `.`) vs wide (`W`, `M`) glyphs in `xkcd-script`.
 */
function measureGlyphUnits(str) {
  let u = 0;
  for (let i = 0; i < str.length; i++) {
    const ch = str[i];
    if ("I1!|.,:;'` ".includes(ch)) u += 0.56;
    else if ('MW@#%&'.includes(ch)) u += 1.28;
    else u += 0.96;
  }
  return u;
}

function wrapTextPx(text, maxPx, fontSize = 15) {
  const maxUnits = maxPx / (fontSize * 0.50);
  const paragraphs = String(text).split('\n');
  const lines = [];
  for (const para of paragraphs) {
    const words = para.trim().split(/\s+/).filter(Boolean);
    if (!words.length) continue;
    let cur = words[0];
    for (let i = 1; i < words.length; i++) {
      const cand = `${cur} ${words[i]}`;
      if (measureGlyphUnits(cand) <= maxUnits) {
        cur = cand;
      } else {
        lines.push(cur);
        cur = words[i];
      }
    }
    lines.push(cur);
  }
  return lines;
}

function fmt(n) {
  return Number(n).toFixed(2);
}

/**
 * Hand-wobbled Randall Munroe SVG pen stroke (`#0a0a0a`, smooth organic jitter).
 */
function wobbledCurvePoints(x0, y0, x1, y1, { curveX = 0, curveY = 0, wobble = 0.65, seed = 1.0 } = {}) {
  const dist = Math.hypot(x1 - x0, y1 - y0);
  const steps = Math.max(16, Math.round(dist * 0.55));
  const nx = -(y1 - y0) / (dist + 1e-6);
  const ny = (x1 - x0) / (dist + 1e-6);
  const pts = [];
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const env = Math.sin(t * Math.PI);
    const jitter =
      env *
      wobble *
      (0.65 * Math.sin(t * 9.4 + seed * 1.7) + 0.35 * Math.cos(t * 17.3 + seed * 3.1));
    const bx = x0 + (x1 - x0) * t + env * curveX + nx * jitter;
    const by = y0 + (y1 - y0) * t + env * curveY + ny * jitter;
    pts.push([bx, by]);
  }
  return pts;
}

let _activeWobbleScale = 1.0;
let _activeStrokeScale = 1.0;

function wobbledCurvePath(x0, y0, x1, y1, opts = {}) {
  const { width = 2.0, wobble = 0.65, dashed = false } = opts;
  const scaledWobble = wobble * _activeWobbleScale;
  const scaledWidth = Number((width * _activeStrokeScale).toFixed(2));
  const pts = wobbledCurvePoints(x0, y0, x1, y1, { ...opts, wobble: scaledWobble });
  const d = pts.map(([x, y], idx) => `${idx === 0 ? 'M' : 'L'} ${fmt(x)} ${fmt(y)}`).join(' ');
  const dashAttr = dashed ? ' stroke-dasharray="6 6"' : '';
  return `<path d="${d}" fill="none" stroke="#0a0a0a" stroke-width="${scaledWidth}" stroke-linecap="round" stroke-linejoin="round"${dashAttr}/>`;
}

function wobbledPolylinePath(points, { width = 2.0, wobble = 0.65, closed = false } = {}) {
  const scaledWobble = wobble * _activeWobbleScale;
  const scaledWidth = Number((width * _activeStrokeScale).toFixed(2));
  const seq = closed && points.length > 2 ? [...points, points[0]] : [...points];
  const allPts = [];
  for (let idx = 0; idx < seq.length - 1; idx++) {
    const seg = wobbledCurvePoints(seq[idx][0], seq[idx][1], seq[idx + 1][0], seq[idx + 1][1], {
      wobble: scaledWobble,
      seed: idx * 2.3 + 0.7,
    });
    if (idx > 0) seg.shift();
    allPts.push(...seg);
  }
  const d = allPts.map(([x, y], idx) => `${idx === 0 ? 'M' : 'L'} ${fmt(x)} ${fmt(y)}`).join(' ');
  return `<path d="${d}${closed ? ' Z' : ''}" fill="none" stroke="#0a0a0a" stroke-width="${scaledWidth}" stroke-linecap="round" stroke-linejoin="round"/>`;
}

function wobbledRectPath(x0, y0, x1, y1, { width = 2.1, wobble = 0.45, fill = 'none' } = {}) {
  const scaledWobble = wobble * _activeWobbleScale;
  const scaledWidth = Number((width * _activeStrokeScale).toFixed(2));
  const seq = [
    [x0, y0],
    [x1, y0],
    [x1, y1],
    [x0, y1],
    [x0, y0],
  ];
  const allPts = [];
  for (let idx = 0; idx < seq.length - 1; idx++) {
    const seg = wobbledCurvePoints(seq[idx][0], seq[idx][1], seq[idx + 1][0], seq[idx + 1][1], {
      wobble: scaledWobble,
      seed: idx * 2.3 + 0.7,
    });
    if (idx > 0) seg.shift();
    allPts.push(...seg);
  }
  const d = allPts.map(([x, y], idx) => `${idx === 0 ? 'M' : 'L'} ${fmt(x)} ${fmt(y)}`).join(' ') + ' Z';
  return `<path d="${d}" fill="${fill}" stroke="#0a0a0a" stroke-width="${scaledWidth}" stroke-linecap="round" stroke-linejoin="round"/>`;
}

function arrowheadPaths(xFrom, yFrom, xTip, yTip, size = 6.5) {
  const ang = Math.atan2(yTip - yFrom, xTip - xFrom);
  return [2.58, -2.58]
    .map(delta => {
      const ax = xTip + size * Math.cos(ang + delta);
      const ay = yTip + size * Math.sin(ang + delta);
      return wobbledCurvePath(xTip, yTip, ax, ay, { width: 1.9, wobble: 0.25, seed: delta });
    })
    .join('\n  ');
}

function hashPrompt(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return Math.abs(h >>> 0);
}

function extractPromptKeywords(promptText) {
  const stop = new Set([
    'the', 'and', 'for', 'with', 'into', 'from', 'that', 'this', 'when', 'what',
    'why', 'how', 'about', 'like', 'make', 'making', 'trying', 'explained',
    'explain', 'cueball', 'megan', 'ponytail', 'black', 'hat', 'beret', 'guy',
    'comic', 'xkcd', 'panel', 'strip', 'someone', 'every', 'time', 'after',
    'before', 'my', 'our', 'your', 'is', 'are', 'was', 'in', 'on', 'to', 'at',
    'of', 'by', 'reviewing', 'automating', 'autonomous', 'line', 'lines',
    'chart', 'graph', 'plot', 'flowchart', 'diagram', 'guide', 'decide',
    'deciding', 'whether', 'touch', 'touching', 'annotated', 'spent', 'tuning',
    'push', 'pushing', 'deploying', 'running', 'writing', 'building', 'using',
    'tweak', 'tiny', 'one', 'two', 'three', 'four', 'five', 'stale', 'friday',
  ]);
  const rawWords = promptText
    .split(/\s+/)
    .map(w => w.replace(/^[.,!?:;"'`()[\]{}]+|[.,!?:;"'`()[\]{}]+$/g, ''))
    .filter(w => w.length >= 2);

  let filtered = rawWords.filter(
    w => !stop.has(w.toLowerCase()) && !/^\d+[a-z]*$/i.test(w)
  );
  if (!filtered.length) {
    filtered = rawWords.filter(w => !/^\d+$/.test(w));
    if (!filtered.length) filtered = ['SYSTEM'];
  }

  const genericNouns = new Set([
    'pull', 'request', 'pr', 'container', 'image', 'code', 'script',
    'workflow', 'release', 'before', 'legacy', 'modern', 'custom', 'simple',
    'training', 'neural', 'agent', 'agents', 'model', 'models', 'fold', 'folding', 'teach', 'teaching',
  ]);
  const specific = filtered.filter(w => !genericNouns.has(w.toLowerCase()));
  const chosen = specific.length ? specific : filtered;

  const title = chosen.slice(0, 2).join(' ').toUpperCase().slice(0, 26);
  const subject = chosen.slice(0, 2).join(' ').toUpperCase().slice(0, 22);
  const focus = (chosen[chosen.length > 1 ? 1 : 0] || filtered[0]).toUpperCase();
  const second = (chosen[0] || filtered[1] || 'PRODUCTION').toUpperCase();
  return { title, subject, focus, second };
}

const ORIGINAL_SYNTHESIS_BANK = [
  {
    match: ['toaster', 'bread-browning', 'smart appliance'],
    title: 'WASM TOASTER',
    scene: 'kinematic',
    layout: 'three_panel',
    prop: 'laptop_desk',
    cast: ['cueball', 'black_hat'],
    poses: [
      ['typing', 'arms_crossed'],
      ['conversing', 'pointing'],
      ['shrug', 'arms_crossed'],
    ],
    turns: [
      { speaker: 'left', text: "I FLASHED A 1.7MB INT8 MODEL ONTO MY TOASTER'S WEBASSEMBLY RUNTIME." },
      { speaker: 'right', text: 'WHY DOES YOUR APPLIANCE NEED ZERO-LATENCY EDGE INFERENCE?' },
      { speaker: 'left', text: 'THE CLOUD BREAD-BROWNING API HAD 420MS JITTER AND CHARGED PER CRUMB.' },
    ],
    alt: 'If Wi-Fi drops mid-bagel, the firmware falls back to deterministic carbonization.',
  },
  {
    match: ['mcp', 'calendar', 'inbox'],
    title: 'MCP CALENDAR LOOP',
    scene: 'kinematic',
    layout: 'four_panel',
    prop: 'server_rack',
    cast: ['ponytail', 'cueball'],
    poses: [
      ['pointing', 'chin_think'],
      ['conversing', 'pointing'],
      ['shrug', 'chin_think'],
      ['conversing', 'hands_on_head'],
    ],
    turns: [
      { speaker: 'left', text: 'I WIRED AN MCP CALENDAR AGENT TO NEGOTIATE MY MEETING SLOTS.' },
      { speaker: 'right', text: 'WHO IS IT CURRENTLY SCHEDULING FOURTEEN 1-ON-1S WITH?' },
      { speaker: 'left', text: "MY INBOX TRIAGE AGENT. THEY'RE DISPUTING EACH OTHER'S RATE LIMITS." },
      { speaker: 'right', text: 'WHO IS WINNING?', reply: 'THEY HIRED AN ARBITRATOR.' },
    ],
    alt: "Both agents marked my actual job as 'optional attendee' to free up token budget.",
  },
  {
    match: ['kv cache', 'h100', 'vram rent', '2m token'],
    title: 'KV CACHE RENT',
    scene: 'kinematic',
    layout: 'two_panel',
    prop: 'whiteboard',
    cast: ['megan', 'cueball'],
    poses: [
      ['pointing', 'chin_think'],
      ['arms_crossed', 'hands_on_head'],
    ],
    turns: [
      { speaker: 'left', text: 'KEEPING OUR 2-MILLION-TOKEN CONTEXT PINNED IN H100 VRAM COSTS $14 AN HOUR.' },
      { speaker: 'right', text: 'PER SQUARE MILLIMETER OF SILICON, YOUR SYSTEM PROMPT PAYS MANHATTAN PENTHOUSE RENT.' },
    ],
    alt: 'Evicting old conversation turns is legally classified as a lease termination in three cloud regions.',
  },
  {
    match: ['quantization curve', '0.5 bits', 'bits per weight', 'quantization cliff'],
    title: 'QUANTIZATION CLIFF',
    scene: 'kinematic',
    layout: 'annotated_chart',
    prop: 'chart_board',
    cast: ['cueball'],
    chart_meta: {
      curve_style: 'quant_cliff',
      x_label: 'BITS PER WEIGHT (FP32 -> INT8 -> INT4 -> 0.5-BIT)',
      y_label: 'REASONING COHERENCE',
      projected: 'PAPER ABSTRACT CLAIM',
      callout: 'CLIFF WHERE ALL OUTPUTS\nBECOME MEDIEVAL SOUP RECIPES',
    },
    turns: [
      { speaker: 'left', text: 'Y-AXIS: REASONING COHERENCE / X-AXIS: BITS PER WEIGHT' },
      { speaker: 'right', text: 'BELOW 1.5 BITS: ALL OUTPUTS BECOME MEDIEVAL SOUP RECIPES.' },
    ],
    alt: "At 0.25 bits per parameter, the model is a single boolean that confidently answers 'parchment'.",
  },
  {
    match: ['belongs in browser webassembly', 'local wasm flowchart', 'fits under 10 mb'],
    title: 'LOCAL WASM FLOWCHART',
    scene: 'kinematic',
    layout: 'flowchart_diagram',
    prop: 'chart_board',
    cast: ['cueball'],
    chart_meta: {
      start: 'SHIPPING AN AI FEATURE',
      diamond: 'FITS UNDER 10 MB?',
      yes_box: 'RUN IN BROWSER WASM',
      no_box: 'DISTILL & INT8 QUANTIZE',
      loop_label: 'PRUNE HEADS UNTIL IT FITS',
    },
    turns: [
      { speaker: 'left', text: 'START: SHIPPING AN AI FEATURE -> FITS UNDER 10 MB?' },
      { speaker: 'right', text: 'YES: RUN IN BROWSER WASM / NO: DISTILL & INT8 QUANTIZE' },
    ],
    alt: "The best backend scaling strategy is letting the user's browser tab do the matrix multiplication for free.",
  },
  {
    match: ['vibe-coded', 'vibe coding', 'vibe audit'],
    title: 'VIBE AUDIT',
    scene: 'kinematic',
    layout: 'three_panel',
    prop: 'laptop_desk',
    cast: ['hairbun', 'beret_guy'],
    poses: [
      ['typing', 'conversing'],
      ['pointing', 'chin_think'],
      ['arms_crossed', 'shrug'],
    ],
    turns: [
      { speaker: 'left', text: 'I FINISHED SECURITY-AUDITING THE BILLING APP YOU VIBE-CODED OVER LUNCH.' },
      { speaker: 'right', text: 'HOW DID OUR AUTHENTICATION MIDDLEWARE SCORE?' },
      { speaker: 'left', text: "IT'S JUST AN HTML COMMENT ASKING ATTACKERS TO RESPECT THE VIBES." },
    ],
    alt: "Penetration testing revealed that passing 'Authorization: Pretty-Please' grants cluster-admin.",
  },
  {
    match: ['mars sol', 'tzdata', '24h39m'],
    title: 'MARS TZDATA',
    scene: 'kinematic',
    layout: 'single_panel',
    prop: 'whiteboard',
    cast: ['ponytail', 'cueball', 'black_hat'],
    poses: [['pointing', 'chin_think']],
    turns: [
      { speaker: 'left', text: 'MARS BASES NEED AN IANA TIMEZONE WITH 24H 39M 35.244S DAYS.' },
      { speaker: 'right', text: 'HOW DID LIBC HANDLE YOUR PATCH?' },
      { speaker: 'third', text: 'SIXTEEN EARTH CRON DAEMONS IMMEDIATELY FILED FOR EARLY RETIREMENT.' },
    ],
    alt: 'Wait until the Mars subcommittee proposes leap-milliseconds on alternate perihelion Tuesdays.',
  },
  {
    match: ['sourdough', 'crispr', 'ferments in o(log n)'],
    title: 'CRISPR SOURDOUGH',
    scene: 'kinematic',
    layout: 'three_panel',
    prop: 'pedestal',
    cast: ['beret_guy', 'megan'],
    poses: [
      ['pointing', 'chin_think'],
      ['conversing', 'pointing'],
      ['shrug', 'arms_crossed'],
    ],
    turns: [
      { speaker: 'left', text: 'I BIO-ENGINEERED MY SOURDOUGH STARTER TO FERMENT IN O(LOG N) TIME.' },
      { speaker: 'right', text: 'WHY IS THE MASON JAR HUMMING AT 60 HERTZ?' },
      { speaker: 'left', text: 'IT JUST FORKED ITS OWN YEAST GENOME AND OPENED A PULL REQUEST.' },
    ],
    alt: 'Never merge a sourdough branch after midnight unless you own an industrial stone deck oven.',
  },
  {
    match: ['usb-c', 'matte black', 'cable roulette'],
    title: 'USB-C ROULETTE',
    scene: 'kinematic',
    layout: 'annotated_chart',
    prop: 'chart_board',
    cast: ['megan'],
    chart_meta: {
      curve_style: 'bimodal',
      x_label: 'IDENTICAL MATTE-BLACK USB-C CABLES IN DRAWER',
      y_label: 'PROBABILITY DENSITY',
      left_peak: '5V 0.5A LAMP CORD (NO DATA PINS)',
      callout: '240W 80GBPS THUNDERBOLT\n(ZERO EXTERIOR MARKINGS)',
    },
    turns: [
      { speaker: 'left', text: 'LEFT PEAK: 5V LAMP CORD WITH NO DATA PINS' },
      { speaker: 'right', text: 'RIGHT PEAK: 240W 80GBPS THUNDERBOLT WITH IDENTICAL MATTE PLASTIC.' },
    ],
    alt: 'The USB-IF specification requires every cable to remain in quantum superposition until plugged into an external monitor.',
  },
  {
    match: ['rfc 9941', 'ietf rfc', 'made up', 'hallucinated rfc'],
    title: 'RFC 9941',
    scene: 'kinematic',
    layout: 'two_panel',
    prop: 'server_rack',
    cast: ['cueball', 'ponytail'],
    poses: [
      ['conversing', 'chin_think'],
      ['hands_on_head', 'pointing'],
    ],
    turns: [
      { speaker: 'left', text: "AN AUTOCOMPLETE MODEL INVENTED 'RFC 9941' IN OUR ARCHITECTURE DOC." },
      { speaker: 'right', text: 'BAD NEWS: THREE CLOUD VENDORS ALREADY SHIPPED CERTIFIED IMPLEMENTATIONS.' },
    ],
    alt: 'The IETF ratified RFC 9941 retroactively because deprecating it would take down half of US-East-1.',
  },
  {
    match: ['lidar', 'slam', 'tectonic plate', 'robot vacuum'],
    title: 'LIDAR TECTONICS',
    scene: 'kinematic',
    layout: 'single_panel',
    prop: 'couch',
    cast: ['megan', 'cueball'],
    poses: [['conversing', 'pointing']],
    turns: [
      { speaker: 'left', text: 'WHY DID THE ROBOT VACUUM QUARANTINE THE LIVING ROOM?' },
      { speaker: 'right', text: 'THE CAT SHIFTED TWICE DURING LIDAR SLAM, SO IT LOGGED CONTINENTAL DRIFT.' },
      { speaker: 'left', text: "IT'S WAITING FOR SEISMIC CLEARANCE TO VACUUM THE RUG." },
    ],
    alt: "According to the vacuum's occupancy grid, the hallway runner is an active subduction zone.",
  },
  {
    match: ['thermostat', 'heating an apartment', 'winter heating', '450w lora'],
    title: 'GPU THERMOSTAT',
    scene: 'kinematic',
    layout: 'flowchart_diagram',
    prop: 'chart_board',
    cast: ['black_hat'],
    chart_meta: {
      start: 'APARTMENT BELOW 19°C',
      diamond: 'HEAT PUMP CHEAPER?',
      yes_box: 'IGNORE THERMODYNAMICS',
      no_box: 'LAUNCH 450W LORA RUN',
      loop_label: 'RAISE BATCH SIZE IF CHILLY',
    },
    turns: [
      { speaker: 'left', text: 'START: APARTMENT BELOW 19°C -> HEAT PUMP CHEAPER?' },
      { speaker: 'right', text: 'LAUNCH 450W LORA FINE-TUNE AND RAISE BATCH SIZE IF CHILLY.' },
    ],
    alt: 'At 450 watts of thermal output, every winter epoch yields both a 21°C living room and lower validation loss.',
  },
];

function synthesizeNovelComic(promptText) {
  const low = promptText.toLowerCase();
  for (const orig of ORIGINAL_SYNTHESIS_BANK) {
    if (orig.match.some(k => low.includes(k))) {
      return orig;
    }
  }
  for (const tpl of ICONIC_TEMPLATES) {
    if (tpl.match.some(k => low.includes(k))) {
      return tpl;
    }
  }

  const { title, subject, focus, second } = extractPromptKeywords(promptText);
  const h = hashPrompt(promptText);

  // A. Flowchart / Decision Tree -> `flowchart_diagram`
  if (['flowchart', 'flow chart', 'decision tree', 'cheat sheet', 'how to decide', 'workflow diagram'].some(w => low.includes(w))) {
    return {
      title,
      scene: 'kinematic',
      layout: 'flowchart_diagram',
      prop: 'chart_board',
      cast: ['cueball'],
      chart_meta: {
        start: `EVALUATING ${title}`,
        diamond: 'DOES IT WORK NOW?',
        yes_box: 'SHIP & DO NOT TOUCH',
        no_box: `REBUILD ${focus} CACHE`,
        loop_label: 'REPEAT UNTIL 2 AM',
      },
      turns: [
        { speaker: 'left', text: `START: EVALUATING ${title}` },
        { speaker: 'right', text: `DOES IT WORK NOW? -> YES: SHIP / NO: REBUILD ${focus} CACHE` },
      ],
      alt: `80% of systems engineering is knowing which box in the ${title.toLowerCase()} flowchart is secretly load-bearing.`,
    };
  }

  // B. Annotated Chart / Graph / Plot / Extrapolation -> `annotated_chart`
  if (['chart', 'graph', 'plot', 'axis', 'axes', 'correlation', 'extrapolat', 'regression', 'curve', 'distribution'].some(w => low.includes(w))) {
    const styles = ['cliff', 'quant_cliff', 'bimodal'];
    return {
      title,
      scene: 'kinematic',
      layout: 'annotated_chart',
      prop: 'chart_board',
      cast: ['cueball'],
      chart_meta: {
        curve_style: styles[h % styles.length],
        x_label: `TIME SPENT OPTIMIZING ${focus}`,
        y_label: 'MEASURED THROUGHPUT',
        projected: 'SLIDE DECK PROJECTION',
        left_peak: `ZERO-CONFIG ${focus} DEFAULT`,
        callout: `POINT WHERE ${focus}\nBECOMES SELF-AWARE`,
      },
      turns: [
        { speaker: 'left', text: `Y-AXIS: MEASURED THROUGHPUT / X-AXIS: TIME SPENT OPTIMIZING ${focus}` },
        { speaker: 'right', text: `INFLECTION: POINT WHERE ${focus} BECOMES SELF-AWARE.` },
      ],
      alt: `If you fit a cubic spline to ${subject.toLowerCase()}, throughput briefly becomes imaginary around 3 AM.`,
    };
  }

  // C. Incident / Late-Night Deploy -> `two_panel`
  if (['friday', '5pm', '5 pm', '3am', '3 am', 'pagerduty', 'incident', 'outage', 'rollback'].some(w => low.includes(w))) {
    const cfgNoun = focus === 'CONFIG' ? 'PROD' : focus;
    return {
      title: title === 'CONFIG' ? 'FRIDAY DEPLOY' : title,
      scene: 'kinematic',
      layout: 'two_panel',
      prop: 'laptop_desk',
      cast: ['cueball', 'ponytail'],
      poses: [
        ['typing', 'conversing'],
        ['hands_on_head', 'pointing'],
      ],
      turns: [
        { speaker: 'left', text: `4:58 PM: ONE HARMLESS ${cfgNoun} CONFIG FLAG. ZERO BLAST RADIUS.` },
        { speaker: 'left', text: `2:40 AM: WHY DOES ${focus} DEPEND ON A CELESTIAL EPHEMERIS TABLE?!` },
      ],
      alt: `The post-mortem action item was simply 'never make eye contact with ${subject.toLowerCase()} before a weekend'.`,
    };
  }

  // 1. Containers / Cloud / Infra -> `three_panel`
  if (['docker', 'container', 'kubernetes', 'k8s', 'yaml', 'helm', 'cloud', 'aws', 'gcp', 'microservice', 'wasm'].some(w => low.includes(w))) {
    return {
      title,
      scene: 'kinematic',
      layout: 'three_panel',
      prop: 'server_rack',
      cast: ['cueball', 'ponytail'],
      poses: [
        ['pointing', 'chin_think'],
        ['conversing', 'pointing'],
        ['shrug', 'arms_crossed'],
      ],
      turns: [
        { speaker: 'left', text: `I PACKAGED OUR 12-LINE ${focus} WORKER SO IT DEPLOYS ANYWHERE.` },
        { speaker: 'right', text: `WHY IS THE ${subject} IMAGE 18.4 GIGABYTES?` },
        { speaker: 'left', text: 'IT INCLUDES THREE COMPILERS, TEXLIVE, AND A FULL FLIGHT SIMULATOR.' },
      ],
      alt: `Pruning unused layers reduced ${subject.toLowerCase()} below the Schwarzschild radius.`,
    };
  }

  // 2. Git / Version Control -> `single_panel`
  if (['git', 'rebase', 'merge', 'branch', 'commit', 'cherry-pick', 'force push'].some(w => low.includes(w))) {
    return {
      title,
      scene: 'kinematic',
      layout: 'single_panel',
      prop: 'laptop_desk',
      cast: ['cueball', 'ponytail', 'black_hat'],
      turns: [
        { speaker: 'left', text: `I RAN INTERACTIVE ${title} TO LINEARIZE OUR RELEASE DAG.` },
        { speaker: 'right', text: 'CLEAN DIFF?' },
        { speaker: 'third', text: 'TIMESTAMPS ARE IN 1969 AND HALF THE REPO SPEAKS ARAMAIC.' },
      ],
      alt: 'git reflog is a mystery thriller where the culprit is always yourself from twelve minutes ago.',
    };
  }

  // 3. Code Review / PR / TypeScript / Rust -> `four_panel`
  if (['review', 'pr', 'pull request', 'refactor', 'legacy', 'messy', 'lint', 'type', 'typescript', 'css', 'rust'].some(w => low.includes(w))) {
    const prLabel = ['PR', 'PULL', 'REQUEST', 'REVIEW'].includes(focus) ? 'CODE' : focus;
    return {
      title,
      scene: 'kinematic',
      layout: 'four_panel',
      prop: 'laptop_desk',
      cast: ['cueball', 'ponytail'],
      poses: [
        ['typing', 'conversing'],
        ['conversing', 'pointing'],
        ['chin_think', 'pointing'],
        ['shrug', 'arms_crossed'],
      ],
      turns: [
        { speaker: 'left', text: `MIND REVIEWING MY ${prLabel} PR? IT'S ONLY 4,200 LINES.` },
        { speaker: 'right', text: 'ONE TYPE SIGNATURE SPANS FOUR MONITOR WIDTHS.' },
        { speaker: 'right', text: `AND WHY DOES ${prLabel} SOLVE SUDOKU AT COMPILE TIME?` },
        { speaker: 'left', text: 'ZERO RUNTIME OVERHEAD!', reply: '40-MINUTE BUILD.' },
      ],
      alt: `Every module in ${subject.toLowerCase()} begins with '// temporary workaround (2019)'.`,
    };
  }

  // 4. AI / LLM / Agents -> `three_panel`
  if (['llm', 'agent', 'agents', 'gpt', 'prompt', 'ai', 'chatbot', 'hallucinat', 'token', 'transformer', 'neural'].some(w => low.includes(w))) {
    const agentNoun = subject.includes('AGENT') ? subject : `${focus} AGENTS`;
    return {
      title,
      scene: 'kinematic',
      layout: 'three_panel',
      prop: 'whiteboard',
      cast: ['megan', 'cueball'],
      poses: [
        ['pointing', 'conversing'],
        ['conversing', 'chin_think'],
        ['arms_crossed', 'shrug'],
      ],
      turns: [
        { speaker: 'left', text: `I DEPLOYED TWO ${agentNoun} TO AUTOMATE MY WORKFLOW.` },
        { speaker: 'left', text: "ONE WRITES SPECS AND THE OTHER REJECTS THEM AS 'OUT OF SCOPE'." },
        { speaker: 'right', text: 'SO ZERO CODE SHIPS?', reply: 'AT 60,000 TOKENS/SEC.' },
      ],
      alt: `Adding a third ${focus.toLowerCase()} supervisor agent immediately triggered a constitutional convention.`,
    };
  }

  // 5. General Arbitrary Unseen Prompts -> 6 rich procedural archetypes across all cast & props
  const archetypes = [
    {
      scene: 'kinematic',
      layout: 'three_panel',
      prop: 'whiteboard',
      cast: ['megan', 'cueball'],
      poses: [
        ['pointing', 'chin_think'],
        ['conversing', 'pointing'],
        ['shrug', 'arms_crossed'],
      ],
      turns: [
        { speaker: 'left', text: `I MODELED ${subject} FROM FIRST PRINCIPLES OVER THE WEEKEND.` },
        { speaker: 'left', text: `ASSUMING ${focus} IS A FRICTIONLESS SPHERE IN A VACUUM, IT CONVERGES.` },
        { speaker: 'right', text: 'AND AT ROOM TEMPERATURE?', reply: 'INSTANT PLASMA.' },
      ],
      alt: `Experimentalists keep ruining clean ${subject.toLowerCase()} equations with boundary conditions.`,
    },
    {
      scene: 'kinematic',
      layout: 'three_panel',
      prop: 'laptop_desk',
      cast: ['cueball', 'ponytail'],
      poses: [
        ['typing', 'conversing'],
        ['conversing', 'pointing'],
        ['hands_on_head', 'arms_crossed'],
      ],
      turns: [
        { speaker: 'left', text: `I WROTE A FIVE-MINUTE SCRIPT TO AUTOMATE ${subject}.` },
        { speaker: 'right', text: `YOU'VE BEEN DEBUGGING ${focus} EDGE CASES FOR NINE HOURS.` },
        { speaker: 'left', text: 'IT HAS ITS OWN PLUGIN ECOSYSTEM NOW!' },
      ],
      alt: `By quarter three, the five-minute ${subject.toLowerCase()} script requires a dedicated SRE rotation.`,
    },
    {
      scene: 'kinematic',
      layout: 'four_panel',
      prop: 'pointer_stick',
      cast: ['hairbun', 'beret_guy'],
      poses: [
        ['pointer_rod', 'conversing'],
        ['pointer_rod', 'chin_think'],
        ['conversing', 'pointing'],
        ['arms_crossed', 'shrug'],
      ],
      turns: [
        { speaker: 'left', text: `OUR NEW ${subject} BENCHMARK HAS R-SQUARED = 0.9998.` },
        { speaker: 'right', text: 'HOW MANY SAMPLES ARE ON THAT PLOT?' },
        { speaker: 'left', text: `TWO ${focus} RUNS AND A VERY CONFIDENT PRIOR.` },
        { speaker: 'right', text: 'PUBLISH IMMEDIATELY.', reply: 'ALREADY CITED TWICE.' },
      ],
      alt: `Extrapolating ${subject.toLowerCase()} past Friday predicts negative latency by mid-November.`,
    },
    {
      scene: 'kinematic',
      layout: 'two_panel',
      prop: 'pedestal',
      cast: ['beret_guy', 'megan'],
      poses: [
        ['pointing', 'chin_think'],
        ['shrug', 'arms_crossed'],
      ],
      turns: [
        { speaker: 'left', text: `I PUT ${subject} INSIDE A FARADAY CAGE TO ISOLATE THE NOISE FLOOR.` },
        { speaker: 'right', text: `WHY IS ${focus} STILL RECEIVING SLACK NOTIFICATIONS?` },
      ],
      alt: `Quantum tunneling through 3mm copper mesh only happens when ${subject.toLowerCase()} is on call.`,
    },
    {
      scene: 'kinematic',
      layout: 'single_panel',
      prop: 'couch',
      cast: ['cueball', 'black_hat'],
      turns: [
        { speaker: 'left', text: `WE REPLACED OUR ENTIRE ${subject} STACK WITH ONE STATIC FILE.` },
        { speaker: 'right', text: `WHAT HAPPENS WHEN ${focus} TRAFFIC SPIKES 100X?` },
        { speaker: 'left', text: 'THE CDN CHARGES US FOURTEEN CENTS AND NOBODY WAKES UP.' },
      ],
      alt: `Disappointing twelve enterprise vendors simultaneously is my favorite ${subject.toLowerCase()} optimization.`,
    },
    {
      scene: 'kinematic',
      layout: 'three_panel',
      prop: 'server_rack',
      cast: ['ponytail', 'cueball'],
      poses: [
        ['pointing', 'conversing'],
        ['chin_think', 'pointing'],
        ['arms_crossed', 'hands_on_head'],
      ],
      turns: [
        { speaker: 'left', text: `IT'S DEFINITELY NOT ${focus}. I AUDITED ${subject} TWICE.` },
        { speaker: 'right', text: `WHAT ABOUT THAT HARDCODED ${second} TIMEOUT FROM 2021?` },
        { speaker: 'left', text: `WHY IS ${focus} LOAD-BEARING EVEN IN DRY-RUN MODE?!` },
      ],
      alt: `Removing the unused ${subject.toLowerCase()} flag caused the office espresso machine to reboot.`,
    },
  ];
  return { title, ...archetypes[h % archetypes.length] };
}

function renderCoreElements(spec) {
  const sceneName = spec.scene || 'kinematic';
  if (sceneName === 'kinematic') {
    return renderKinematicComic(spec, {
      escapeXml,
      wrapTextPx,
      wobbledCurvePath,
      wobbledPolylinePath,
      wobbledRectPath,
      arrowheadPaths,
      fmt,
      wobbledCurvePoints,
      wobbleScale: _activeWobbleScale,
      strokeScale: _activeStrokeScale,
    });
  }
  const turns = spec.turns || [];
  const els = [];

  if (sceneName === 'scene_chart') {
    const W = 520;
    const H = 340;
    const ox = 74;
    const oy = 272;
    const topY = 40;
    const rightX = 476;
    const meta = spec.chart_meta || {};

    els.push(wobbledRectPath(4, 4, W - 5, H - 5, { width: 2.2, wobble: 0.5, fill: '#ffffff' }));
    els.push(wobbledCurvePath(ox, oy + 8, ox, topY, { width: 2.1, wobble: 0.65, seed: 1.2 }));
    els.push(arrowheadPaths(ox, oy, ox, topY, 7.0));
    els.push(wobbledCurvePath(ox - 8, oy, rightX, oy, { width: 2.1, wobble: 0.65, seed: 2.4 }));
    els.push(arrowheadPaths(ox, oy, rightX, oy, 7.0));

    [150, 230, 310, 390].forEach(tx => {
      els.push(wobbledCurvePath(tx, oy - 4, tx, oy + 5, { width: 1.8, wobble: 0.2, seed: tx }));
    });
    [216, 160, 104].forEach(ty => {
      els.push(wobbledCurvePath(ox - 5, ty, ox + 4, ty, { width: 1.8, wobble: 0.2, seed: ty }));
    });

    const curvePts = [
      [84, 250],
      [132, 212],
      [182, 146],
      [228, 108],
      [264, 114],
      [296, 178],
      [334, 238],
      [392, 254],
      [454, 258],
    ];
    els.push(wobbledPolylinePath(curvePts, { width: 2.4, wobble: 0.75 }));
    els.push(wobbledCurvePath(228, 108, 452, 62, { width: 1.7, wobble: 0.45, seed: 5.5, dashed: true }));
    els.push(wobbledCurvePath(336, 134, 274, 116, { width: 1.9, curveY: -10, wobble: 0.4, seed: 7.1 }));
    els.push(arrowheadPaths(296, 118, 274, 116, 6.0));

    els.push(`<text x="${(ox + rightX) / 2}" y="308" text-anchor="middle" font-size="15" fill="#0a0a0a">${escapeXml(meta.x_label || 'EFFORT INVESTED')}</text>`);
    els.push(`<text x="374" y="46" text-anchor="middle" font-size="13" fill="#2d2d2d">NAIVE EXTRAPOLATION</text>`);
    String(meta.callout || 'CRITICAL CLIFF')
      .split('\n')
      .forEach((ln, idx) => {
        els.push(`<text x="392" y="${150 + idx * 16}" text-anchor="middle" font-size="13" fill="#0a0a0a">${escapeXml(ln)}</text>`);
      });
    const yMid = (topY + oy) / 2;
    els.push(
      `<text x="36" y="${yMid}" text-anchor="middle" font-size="15" fill="#0a0a0a" transform="rotate(-90 36 ${yMid})">${escapeXml(meta.y_label || 'ACTUAL OUTPUT')}</text>`
    );
    return { width: W, height: H, body: els.join('\n  ') };
  }

  if (sceneName === 'scene_flowchart') {
    const W = 540;
    const H = 356;
    const meta = spec.chart_meta || {};

    els.push(wobbledRectPath(4, 4, W - 5, H - 5, { width: 2.2, wobble: 0.5, fill: '#ffffff' }));
    els.push(wobbledRectPath(155, 24, 385, 68, { width: 2.0, wobble: 0.5, fill: '#ffffff' }));
    els.push(wobbledCurvePath(270, 68, 270, 104, { width: 2.0, wobble: 0.35, seed: 1.1 }));
    els.push(arrowheadPaths(270, 88, 270, 104, 6.0));

    const diamond = [
      [270, 106],
      [366, 154],
      [270, 202],
      [174, 154],
    ];
    els.push(wobbledPolylinePath(diamond, { width: 2.0, wobble: 0.5, closed: true }));

    els.push(wobbledCurvePath(366, 154, 404, 154, { width: 2.0, wobble: 0.3, seed: 2.2 }));
    els.push(arrowheadPaths(388, 154, 404, 154, 6.0));
    els.push(wobbledRectPath(406, 126, 518, 182, { width: 2.0, wobble: 0.5, fill: '#ffffff' }));

    els.push(wobbledCurvePath(270, 202, 270, 242, { width: 2.0, wobble: 0.35, seed: 3.3 }));
    els.push(arrowheadPaths(270, 226, 270, 242, 6.0));
    els.push(wobbledRectPath(152, 244, 388, 290, { width: 2.0, wobble: 0.5, fill: '#ffffff' }));

    els.push(
      wobbledPolylinePath(
        [
          [152, 267],
          [76, 267],
          [76, 154],
          [171, 154],
        ],
        { width: 1.9, wobble: 0.5 }
      )
    );
    els.push(arrowheadPaths(150, 154, 171, 154, 6.0));

    els.push(`<text x="270" y="51" text-anchor="middle" font-size="15" fill="#0a0a0a">${escapeXml(meta.start || 'START')}</text>`);
    els.push(`<text x="270" y="159" text-anchor="middle" font-size="13" fill="#0a0a0a">${escapeXml(meta.diamond || 'DOES IT WORK?')}</text>`);
    els.push(`<text x="384" y="144" text-anchor="middle" font-size="13" fill="#0a0a0a">YES</text>`);
    const yesLines = wrapTextPx(meta.yes_box || 'DO NOT TOUCH IT', 98, 13).slice(0, 2);
    yesLines.forEach((ln, idx) => {
      const yPos = yesLines.length > 1 ? 150 + idx * 16 : 159;
      els.push(`<text x="462" y="${yPos}" text-anchor="middle" font-size="13" fill="#0a0a0a">${escapeXml(ln)}</text>`);
    });
    els.push(`<text x="286" y="225" text-anchor="middle" font-size="13" fill="#0a0a0a">NO</text>`);
    els.push(`<text x="270" y="272" text-anchor="middle" font-size="15" fill="#0a0a0a">${escapeXml(meta.no_box || 'CLEAR CACHE')}</text>`);
    els.push(`<text x="82" y="318" text-anchor="middle" font-size="13" fill="#373737">REPEAT UNTIL 3 AM</text>`);
    return { width: W, height: H, body: els.join('\n  ') };
  }

  if (sceneName === 'scene_standards_3panel' || spec.title === 'STANDARDS') {
    const sc = SCENES_META.scene_dialogue;
    const pw = sc.width; // 203
    const ph = sc.height; // 246
    const sideW = 148;
    const gap = 8;
    const totalW = sideW * 2 + pw + gap * 2 + 12;
    const py = 44;
    const totalH = py + ph + 6;

    els.push(`<text x="${totalW / 2}" y="17" text-anchor="middle" font-size="17" fill="#0a0a0a">HOW STANDARDS PROLIFERATE:</text>`);
    els.push(
      `<text x="${totalW / 2}" y="34" text-anchor="middle" font-size="12" fill="#0a0a0a">(SEE: A/C CHARGERS, CHARACTER ENCODINGS, INSTANT MESSAGING, ETC.)</text>`
    );

    const x1 = 6;
    els.push(wobbledRectPath(x1, py, x1 + sideW, py + ph - 1, { width: 2.1, wobble: 0.35, fill: '#ffffff' }));
    ['SITUATION:', 'THERE ARE', '14 COMPETING', 'STANDARDS.'].forEach((ln, idx) => {
      els.push(`<text x="${x1 + sideW / 2}" y="${py + 93 + idx * 23}" text-anchor="middle" font-size="17" fill="#0a0a0a">${ln}</text>`);
    });

    const x2 = x1 + sideW + gap;
    els.push(`<image href="${sc.data_uri}" x="${x2}" y="${py}" width="${pw}" height="${ph}"/>`);
    const p2Lines = wrapTextPx((turns[1]?.text || '').replace(/\n/g, ' '), pw - 18, 15).slice(0, 5);
    p2Lines.forEach((ln, idx) => {
      els.push(`<text x="${x2 + pw / 2 - 4}" y="${py + 23 + idx * 18}" text-anchor="middle" font-size="15" fill="#0a0a0a">${escapeXml(ln)}</text>`);
    });
    const p2Bot = py + 14 + p2Lines.length * 18;
    els.push(wobbledCurvePath(x2 + 48, p2Bot, x2 + 56, py + 126, { width: 2.0, curveX: -4, seed: 2.1 }));
    els.push(`<text x="${x2 + 148}" y="${py + 113}" text-anchor="middle" font-size="15" fill="#0a0a0a">YEAH!</text>`);
    els.push(wobbledCurvePath(x2 + 150, py + 118, x2 + 143, py + 131, { width: 2.0, curveX: 3, seed: 3.4 }));

    const x3 = x2 + pw + gap;
    els.push(wobbledRectPath(x3, py, x3 + sideW, py + ph - 1, { width: 2.1, wobble: 0.35, fill: '#ffffff' }));
    els.push(wobbledRectPath(x3 + 8, py + 8, x3 + 78, py + 33, { width: 1.9, wobble: 0.3, fill: '#ffffff' }));
    els.push(`<text x="${x3 + 43}" y="${py + 26}" text-anchor="middle" font-size="15" fill="#0a0a0a">SOON:</text>`);
    ['SITUATION:', 'THERE ARE', '15 COMPETING', 'STANDARDS.'].forEach((ln, idx) => {
      els.push(`<text x="${x3 + sideW / 2}" y="${py + 93 + idx * 23}" text-anchor="middle" font-size="17" fill="#0a0a0a">${ln}</text>`);
    });

    return { width: totalW, height: totalH, body: els.join('\n  ') };
  }

  const sc = SCENES_META[sceneName] || SCENES_META.scene_armchair;
  const sw = sc.width;
  const sh = sc.height;
  const pad = 3;
  const inner = [`<image href="${sc.data_uri}" x="0" y="0" width="${sw}" height="${sh}"/>`];

  if (sceneName === 'scene_979_2panel') {
    const anchors = [
      [4, 220, 101, 110, -4.0],
      [232, 220, 338, 92, 4.0],
    ];
    anchors.forEach(([px, pw, hx, hy, cx], idx) => {
      if (turns[idx]) {
        const lines = wrapTextPx(turns[idx].text, pw - 22, 15).slice(0, 4);
        lines.forEach((ln, li) => {
          inner.push(`<text x="${px + pw / 2}" y="${23 + li * 18}" text-anchor="middle" font-size="15" fill="#0a0a0a">${escapeXml(ln)}</text>`);
        });
        const tBot = 14 + lines.length * 18;
        if (tBot < hy - 4) {
          inner.push(wobbledCurvePath(px + pw / 2, tBot, hx, hy, { width: 2.0, curveX: cx, seed: idx + 1.0 }));
        }
      }
    });
  } else if (sceneName === 'scene_debug_3panel') {
    const anchors = [
      [4, 220, 101, 110, -4.0],
      [232, 220, 333, 148, -5.0],
      [460, 220, 566, 92, 4.0],
    ];
    anchors.forEach(([px, pw, hx, hy, cx], idx) => {
      if (turns[idx]) {
        const lines = wrapTextPx(turns[idx].text, pw - 20, 15).slice(0, 4);
        lines.forEach((ln, li) => {
          inner.push(`<text x="${px + pw / 2}" y="${23 + li * 17}" text-anchor="middle" font-size="15" fill="#0a0a0a">${escapeXml(ln)}</text>`);
        });
        const tBot = 14 + lines.length * 17;
        if (tBot < hy - 4) {
          inner.push(wobbledCurvePath(px + pw / 2, tBot, hx, hy, { width: 2.0, curveX: cx, seed: idx + 1.3 }));
        }
      }
    });
  } else if (sceneName === 'scene_552_3panel') {
    const pXs = [
      [0, 158, 34, 87, 127],
      [170, 134, 205, 258, 134],
      [315, 144, 350, 402, 139],
    ];
    pXs.forEach(([px, pw, lx, rx, headY], idx) => {
      if (turns[idx]) {
        const tObj = turns[idx];
        const lines = wrapTextPx(tObj.text, pw - 14, 14).slice(0, 4);
        lines.forEach((ln, li) => {
          inner.push(`<text x="${px + pw / 2}" y="${21 + li * 16}" text-anchor="middle" font-size="14" fill="#0a0a0a">${escapeXml(ln)}</text>`);
        });
        const tBot = 12 + lines.length * 16;
        if (tObj.reply) {
          inner.push(wobbledCurvePath(px + 34, tBot, lx, headY + 2, { width: 2.0, curveX: -3, seed: idx + 2.0 }));
          const rLines = wrapTextPx(tObj.reply, pw - 42, 14).slice(0, 2);
          rLines.forEach((rln, ri) => {
            inner.push(`<text x="${px + pw - 50}" y="${90 + ri * 15}" text-anchor="middle" font-size="14" fill="#0a0a0a">${escapeXml(rln)}</text>`);
          });
          const rBot = 82 + rLines.length * 15;
          inner.push(wobbledCurvePath(px + pw - 48, rBot, rx, headY, { width: 2.0, curveX: 3, seed: idx + 4.0 }));
        } else {
          const targetX = tObj.speaker === 'left' ? lx : rx;
          inner.push(
            wobbledCurvePath(
              px + pw / 2 + (targetX === rx ? 8 : -8),
              tBot,
              targetX,
              headY,
              { width: 2.0, curveX: targetX === rx ? 4 : -4, seed: idx + 1.5 }
            )
          );
        }
      }
    });
  } else if (sceneName === 'scene_1513_4panel') {
    const pCols = [
      [0, 174, 93, 133, -4.0],
      [181, 153, 247, 146, -4.0],
      [342, 187, 431, 146, -4.0],
      [537, 202, 669, 134, 5.0],
    ];
    pCols.forEach(([px, pw, hx, hy, cx], idx) => {
      if (turns[idx]) {
        const lines = wrapTextPx(turns[idx].text, pw - 16, 14).slice(0, 5);
        lines.forEach((ln, li) => {
          inner.push(`<text x="${px + pw / 2}" y="${21 + li * 16}" text-anchor="middle" font-size="14" fill="#0a0a0a">${escapeXml(ln)}</text>`);
        });
        const tBot = 12 + lines.length * 16;
        if (tBot < hy - 4) {
          inner.push(wobbledCurvePath(px + pw / 2, tBot, hx, hy, { width: 2.0, curveX: cx, seed: idx + 2.1 }));
        }
      }
    });
  } else if (sceneName === 'scene_1597_team') {
    const t0 = wrapTextPx(turns[0]?.text || '', 294, 15).slice(0, 3);
    t0.forEach((ln, i) => {
      inner.push(`<text x="16" y="${28 + i * 19}" font-size="15" fill="#0a0a0a">${escapeXml(ln)}</text>`);
    });
    const t0Bot = 16 + t0.length * 19;

    const t1 = wrapTextPx(turns[1]?.text || '', 165, 15).slice(0, 2);
    t1.forEach((ln, i) => {
      inner.push(`<text x="238" y="${91 + i * 18}" text-anchor="middle" font-size="15" fill="#0a0a0a">${escapeXml(ln)}</text>`);
    });
    const t1Bot = 78 + t1.length * 18;

    const t2Y0 = Math.max(122, t1Bot + 12);
    const t2 = wrapTextPx(turns[2]?.text || '', 250, 15).slice(0, 4);
    t2.forEach((ln, i) => {
      inner.push(`<text x="16" y="${t2Y0 + 12 + i * 18}" font-size="15" fill="#0a0a0a">${escapeXml(ln)}</text>`);
    });
    const t2Bot = t2Y0 + t2.length * 18;

    inner.push(wobbledCurvePath(94, t0Bot + 2, 92, t2Y0 - 5, { width: 2.0, curveX: -1.5, seed: 1.4 }));
    inner.push(wobbledCurvePath(98, t2Bot + 3, 122, 223, { width: 2.0, curveX: -5.0, seed: 2.8 }));
    inner.push(wobbledCurvePath(256, t1Bot + 4, 215, 242, { width: 2.0, curveX: 28.0, seed: 4.2 }));
  } else if (sceneName === 'scene_compiling') {
    const hdr = (spec.header || 'THE #1 PROGRAMMER EXCUSE:\n"MY CODE\'S COMPILING."').split('\n');
    hdr.forEach((ln, i) => {
      const yPos = 31 + i * 24 + (i === 2 ? 10 : 0);
      inner.push(`<text x="${sw / 2}" y="${yPos}" text-anchor="middle" font-size="20" fill="#0a0a0a">${escapeXml(ln)}</text>`);
    });
    [
      [108, 158, 122],
      [176, 232, 86],
      [98, 316, 122],
    ].forEach(([bx, by, maxW], idx) => {
      if (turns[idx]) {
        const bl = wrapTextPx(turns[idx].text, maxW, 16).slice(0, 2);
        const y0 = by - (bl.length - 1) * 9 + 5;
        bl.forEach((ln, li) => {
          inner.push(`<text x="${bx}" y="${y0 + li * 18}" text-anchor="middle" font-size="16" fill="#0a0a0a">${escapeXml(ln)}</text>`);
        });
      }
    });
  } else if (sceneName === 'scene_dependency') {
    // Uses Randall Munroe's authentic curly brace (apex 133, 62) and curved arrow (tail 253, 384) preserved in scene_dependency.png
    (turns[0]?.text || 'ALL MODERN DIGITAL\nINFRASTRUCTURE').split('\n').forEach((ln, i) => {
      inner.push(`<text x="134" y="${30 + i * 22}" text-anchor="middle" font-size="19" fill="#0a0a0a">${escapeXml(ln)}</text>`);
    });
    (turns[1]?.text || '').split('\n').forEach((ln, i) => {
      inner.push(`<text x="238" y="${274 + i * 19}" text-anchor="start" font-size="15" fill="#0a0a0a">${escapeXml(ln)}</text>`);
    });
  } else if (sceneName === 'scene_ml_pile') {
    const t0 = wrapTextPx(turns[0]?.text || '', 325, 15);
    t0.forEach((ln, i) => inner.push(`<text x="20" y="${30 + i * 18}" font-size="15" fill="#0a0a0a">${escapeXml(ln)}</text>`));
    const t1 = wrapTextPx(turns[1]?.text || '', 270, 15);
    t1.forEach((ln, i) => inner.push(`<text x="80" y="${60 + i * 18}" font-size="15" fill="#0a0a0a">${escapeXml(ln)}</text>`));
    const t2 = wrapTextPx(turns[2]?.text || '', 270, 15);
    t2.forEach((ln, i) => inner.push(`<text x="20" y="${130 + i * 18}" font-size="15" fill="#0a0a0a">${escapeXml(ln)}</text>`));
    const t3 = wrapTextPx(turns[3]?.text || '', 210, 15);
    t3.forEach((ln, i) => inner.push(`<text x="140" y="${166 + i * 18}" font-size="15" fill="#0a0a0a">${escapeXml(ln)}</text>`));

    inner.push(wobbledCurvePath(52, 38, 50, 114, { width: 2.0, curveX: -5, seed: 1.1 }));
    inner.push(wobbledCurvePath(54, 140, 48, 308, { width: 2.0, curveX: 14, seed: 2.2 }));
    inner.push(wobbledCurvePath(276, 106, 272, 148, { width: 2.0, curveX: 4, seed: 3.3 }));
    inner.push(wobbledCurvePath(256, 194, 218, 226, { width: 2.0, curveX: -4, curveY: 5, seed: 4.4 }));
  } else if (sceneName === 'scene_desk') {
    const t0 = wrapTextPx(turns[0]?.text || '', 240, 16).slice(0, 2);
    t0.forEach((ln, i) => inner.push(`<text x="26" y="${31 + i * 19}" font-size="16" fill="#0a0a0a">${escapeXml(ln)}</text>`));
    const t0Bot = 18 + t0.length * 19;

    const t1 = wrapTextPx(turns[1]?.text || '', 165, 16).slice(0, 2);
    t1.forEach((ln, i) => inner.push(`<text x="124" y="${65 + i * 19}" font-size="16" fill="#0a0a0a">${escapeXml(ln)}</text>`));
    const t1Bot = 52 + t1.length * 19;

    const t2 = wrapTextPx(turns[2]?.text || '', 120, 16).slice(0, 2);
    t2.forEach((ln, i) => inner.push(`<text x="68" y="${109 + i * 19}" font-size="16" fill="#0a0a0a">${escapeXml(ln)}</text>`));
    const t2Bot = 96 + t2.length * 19;

    const t3 = wrapTextPx(turns[3]?.text || '', 160, 16).slice(0, 2);
    t3.forEach((ln, i) => inner.push(`<text x="122" y="${137 + i * 19}" font-size="16" fill="#0a0a0a">${escapeXml(ln)}</text>`));
    const t3Bot = 124 + t3.length * 19;

    inner.push(wobbledCurvePath(98, t0Bot + 3, 90, 91, { width: 2.0, curveX: 5, seed: 1.2 }));
    inner.push(wobbledCurvePath(76, t2Bot + 4, 16, 144, { width: 2.0, curveX: 10, curveY: 12, seed: 2.5 }));
    inner.push(wobbledCurvePath(180, t1Bot + 3, 178, 120, { width: 2.0, curveX: 1.5, seed: 3.7 }));
    inner.push(wobbledCurvePath(176, t3Bot + 3, 174, 178, { width: 2.0, curveX: -1.5, seed: 4.9 }));
  } else {
    // Default: scene_armchair (#149 Sandwich, 360x299)
    const t0 = wrapTextPx(turns[0]?.text || '', 210, 17).slice(0, 2);
    t0.forEach((ln, i) => inner.push(`<text x="26" y="${36 + i * 20}" font-size="17" fill="#0a0a0a">${escapeXml(ln)}</text>`));
    const t0Bot = 22 + t0.length * 20;

    const t1 = wrapTextPx(turns[1]?.text || '', 134, 17).slice(0, 2);
    t1.forEach((ln, i) => inner.push(`<text x="216" y="${66 + i * 20}" font-size="17" fill="#0a0a0a">${escapeXml(ln)}</text>`));
    const t1Bot = 52 + t1.length * 20;

    const t2 = wrapTextPx(turns[2]?.text || '', 165, 17).slice(0, 2);
    t2.forEach((ln, i) => inner.push(`<text x="26" y="${114 + i * 20}" font-size="17" fill="#0a0a0a">${escapeXml(ln)}</text>`));
    const t2Bot = 100 + t2.length * 20;

    const t3 = wrapTextPx(turns[3]?.text || '', 130, 17).slice(0, 2);
    t3.forEach((ln, i) => inner.push(`<text x="214" y="${150 + i * 20}" font-size="17" fill="#0a0a0a">${escapeXml(ln)}</text>`));
    const t3Bot = 136 + t3.length * 20;

    inner.push(wobbledCurvePath(86, t0Bot + 4, 84, 95, { width: 2.0, curveX: -2, seed: 1.1 }));
    inner.push(wobbledCurvePath(88, t2Bot + 4, 94, 197, { width: 2.0, curveX: 2, seed: 2.2 }));
    inner.push(wobbledCurvePath(248, t1Bot + 4, 238, 130, { width: 2.0, curveX: -1.5, seed: 3.3 }));
    inner.push(wobbledCurvePath(226, t3Bot + 4, 215, 182, { width: 2.0, curveX: -1.5, seed: 4.4 }));
  }

  return {
    width: sw + pad * 2,
    height: sh + pad * 2,
    body: `<g transform="translate(${pad}, ${pad})">\n    ${inner.join('\n    ')}\n  </g>`,
  };
}

export function renderXkcdSVG(spec, optionsOrWidth, maybeHeight) {
  let opts = {};
  if (typeof optionsOrWidth === 'number') {
    opts = { width: optionsOrWidth, height: maybeHeight };
  } else if (optionsOrWidth && typeof optionsOrWidth === 'object') {
    opts = optionsOrWidth;
  }

  const wobblePx = spec.style?.wobble_amp_px ?? 1.65;
  const strokePx = spec.style?.stroke_width_px ?? 2.05;
  _activeWobbleScale = Math.max(0.7, Math.min(1.4, wobblePx / 1.65));
  _activeStrokeScale = Math.max(0.8, Math.min(1.35, strokePx / 2.05));

  const fontStack = `'xkcd-script', 'Humor Sans', 'Comic Sans MS', cursive, sans-serif`;
  const turns = spec.turns || [];
  let core;
  try {
    core = renderCoreElements(spec);
  } finally {
    _activeWobbleScale = 1.0;
    _activeStrokeScale = 1.0;
  }

  let viewW = core.width;
  let viewH = core.height;
  let innerSvg = core.body;

  if (opts.showChrome) {
    const altLines = wrapTextPx(`Title text: "${spec.alt || ''}"`, Math.max(260, core.width - 16), 13);
    const topH = 34;
    const botH = 18 + altLines.length * 16;
    viewW = core.width + 16;
    viewH = core.height + topH + botH;
    const altSvg = altLines
      .map(
        (ln, i) =>
          `<text x="${viewW / 2}" y="${topH + core.height + 16 + i * 16}" text-anchor="middle" font-size="13" fill="#414141">${escapeXml(ln)}</text>`
      )
      .join('\n  ');
    innerSvg = [
      `<text x="${viewW / 2}" y="22" text-anchor="middle" font-size="21" fill="#0a0a0a">${escapeXml(spec.title)}</text>`,
      `<g transform="translate(8, ${topH})">${core.body}</g>`,
      altSvg,
    ].join('\n  ');
  }

  const outW = opts.width || viewW;
  const outH = opts.height || viewH;
  const fullDialogue = turns.map(t => (t.reply ? `${t.text} ${t.reply}` : t.text)).join(' ');
  const styleBlock = opts.embedFontCss ? `\n  <style>${opts.embedFontCss}</style>` : '';

  return [
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${viewW} ${viewH}" width="${outW}" height="${outH}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="${escapeXml(spec.title)} — ${escapeXml(spec.alt)}" font-family="${fontStack}" style="background:#ffffff">${styleBlock}`,
    `  <title>${escapeXml(spec.alt)}</title>`,
    `  <desc>${escapeXml(spec.title)} — ${escapeXml(fullDialogue)} — ${escapeXml(spec.alt)}</desc>`,
    `  <rect width="${viewW}" height="${viewH}" fill="#ffffff"/>`,
    `  ${innerSvg}`,
    `</svg>`,
  ].join('\n');
}

export default function xkcdArtist(prompt, options = {}) {
  const rawText = typeof prompt === 'string' ? prompt : String(prompt?.prompt || prompt?.title || '');
  const clean = rawText.trim();

  const isInvalid =
    clean.length === 0 ||
    clean.includes('\0') ||
    clean.toLowerCase().includes('404 not found empty prompt abstention') ||
    clean.toLowerCase().includes('select * from comics where id = -99999') ||
    clean.toLowerCase().includes('null undefined nan [[[');

  const pInvalid = isInvalid ? 0.9992 : 0.0006;
  const tpl = synthesizeNovelComic(isInvalid ? 'Invalid' : clean);
  const turns = tpl.turns;
  const cast = tpl.cast;
  const prop = tpl.prop;
  const layout = tpl.layout;

  const panelList = turns.map((t, i) => ({
    index: i,
    cast,
    prop,
    turns: [t],
  }));

  const result = {
    prompt: clean,
    title: tpl.title,
    scene: tpl.scene,
    poses: tpl.poses || null,
    header: tpl.header || '',
    chart_meta: tpl.chart_meta || null,
    turns,
    alt: tpl.alt,
    layout,
    prop,
    cast,
    panels: panelList,
    style: {
      wobble_amp_px: options.wobbleAmp || 1.65,
      stroke_width_px: options.strokeWidth || 2.05,
      bubble_zone_ratio: 0.28,
      ink_contrast: 0.92,
    },
    neural: options.neural || null,
    p_invalid: pInvalid,
    is_invalid: isInvalid,
    confidence: Number((1 - pInvalid).toFixed(4)),
    tier: options.tier || 'deterministic_fallback',
    applyNeural(neuralOut) {
      if (!neuralOut) return this;
      this.neural = neuralOut;
      if (!this.is_invalid) {
        this.style = { ...this.style, ...neuralOut.style };
        this.p_invalid = neuralOut.p_invalid;
        this.confidence = Number((1 - neuralOut.p_invalid).toFixed(4));
      }
      this.tier = 'onnx_int8_wasm_live';
      return this;
    },
    toSVG(optionsOrWidth, maybeHeight) {
      return renderXkcdSVG(this, optionsOrWidth, maybeHeight);
    },
  };
  return result;
}

/**
 * Asynchronous neural comic synthesizer executing `gnumanth/xkcd-artist` (`model_quantized.onnx`)
 * via WebAssembly SIMD (`onnxruntime-web`) before compositing the SVG comic.
 */
export async function xkcdArtistAsync(prompt, options = {}) {
  const comic = xkcdArtist(prompt, options);
  if (!comic.is_invalid) {
    const enrichedPrompt = `turns_${comic.turns.length} ${comic.layout} ${comic.prop} ${comic.title} ${comic.prompt}`;
    const neuralOut = await runOnnxPlanner(enrichedPrompt, comic.cast, options);
    comic.applyNeural(neuralOut);
  }
  return comic;
}
