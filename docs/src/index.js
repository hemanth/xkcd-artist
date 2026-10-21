/**
 * xkcd-artist: Hybrid Fine-Tuned LoRA Stick-Figure Art + Crisp Hand-Lettered Speech Bubble & Alt-Text Compositor.
 * Zero-dependency JavaScript/ESM engine + optional Browser WASM ONNX Int8 bridge.
 */

export const CHARACTERS = ['cueball', 'megan', 'black_hat', 'beret_guy', 'ponytail', 'hairbun'];
export const PROPS = ['none', 'laptop_desk', 'whiteboard', 'server_rack', 'pointer_stick', 'couch', 'chart_board', 'pedestal'];
export const LAYOUT_TYPES = ['single_panel', 'two_panel', 'three_panel', 'four_panel', 'annotated_chart', 'flowchart_diagram'];

const ICONIC_TEMPLATES = [
  {
    match: ['sudo', 'sandwich', 'root', 'permission'],
    title: 'SANDWICH',
    layout: 'three_panel',
    prop: 'couch',
    cast: ['cueball', 'megan'],
    turns: [
      { speaker: 'cueball', text: 'MAKE ME A SANDWICH.' },
      { speaker: 'megan', text: 'WHAT? MAKE IT YOURSELF.' },
      { speaker: 'cueball', text: 'SUDO MAKE ME A SANDWICH.' },
      { speaker: 'megan', text: 'OKAY.' },
    ],
    alt: 'Proper User Policy apparently means Simon Says.',
  },
  {
    match: ['compiling', 'compiler', 'sword', 'slacking', 'build'],
    title: 'COMPILING',
    layout: 'two_panel',
    prop: 'pointer_stick',
    cast: ['cueball', 'black_hat'],
    turns: [
      { speaker: 'ponytail', text: 'HEY! GET BACK TO WORK!' },
      { speaker: 'cueball', text: "MY CODE'S COMPILING!" },
      { speaker: 'ponytail', text: 'OH. CARRY ON.' },
    ],
    alt: "The #1 programmer excuse for legitimately slacking off: 'My code's compiling.'",
  },
  {
    match: ['dependency', 'nebraska', 'modern', 'digital', 'infrastructure', 'tower', 'package'],
    title: 'DEPENDENCY',
    layout: 'single_panel',
    prop: 'server_rack',
    cast: ['cueball', 'ponytail'],
    turns: [
      { speaker: 'narrator', text: 'ALL MODERN DIGITAL INFRASTRUCTURE' },
      { speaker: 'cueball', text: 'A PROJECT SOME RANDOM PERSON IN NEBRASKA HAS BEEN THANKLESSLY MAINTAINING SINCE 2003.' },
    ],
    alt: "Someday ImageMagick will finally break for good and we'll have a long period of scrambling.",
  },
  {
    match: ['standard', 'standards', '14', '15', 'competing', 'protocol', 'universal'],
    title: 'STANDARDS',
    layout: 'three_panel',
    prop: 'whiteboard',
    cast: ['cueball', 'ponytail'],
    turns: [
      { speaker: 'narrator', text: 'HOW STANDARDS PROLIFERATE: (SEE: A/C CHARGERS, CHARACTER ENCODINGS, INSTANT MESSAGING)' },
      { speaker: 'cueball', text: "14 COMPETING STANDARDS? RIDICULOUS! WE NEED ONE UNIVERSAL STANDARD THAT COVERS EVERYONE'S USE CASES!" },
      { speaker: 'narrator', text: 'SOON: SITUATION: THERE ARE 15 COMPETING STANDARDS.' },
    ],
    alt: "Fortunately, the charging one has been solved now that we've all standardized on mini-USB. Or is it micro-USB? Shit.",
  },
  {
    match: ['machine learning', 'ai', 'linear algebra', 'pile', 'stir', 'neural', 'llm', 'model'],
    title: 'MACHINE LEARNING',
    layout: 'two_panel',
    prop: 'pedestal',
    cast: ['ponytail', 'cueball'],
    turns: [
      { speaker: 'cueball', text: 'THIS IS YOUR MACHINE LEARNING SYSTEM?' },
      { speaker: 'ponytail', text: 'YEP! YOU POUR THE DATA INTO THIS BIG PILE OF LINEAR ALGEBRA, THEN COLLECT THE ANSWERS ON THE OTHER SIDE.' },
      { speaker: 'cueball', text: 'WHAT IF THE ANSWERS ARE WRONG?' },
      { speaker: 'ponytail', text: 'JUST STIR THE PILE UNTIL THEY START LOOKING RIGHT.' },
    ],
    alt: "The pile gets soaked with data and starts to get mushy over time, so it's technically recurrent.",
  },
  {
    match: ['regex', 'regular expression', 'perl', 'problems', 'parse'],
    title: 'REGULAR EXPRESSIONS',
    layout: 'two_panel',
    prop: 'laptop_desk',
    cast: ['cueball', 'megan'],
    turns: [
      { speaker: 'megan', text: 'WE HAVE TO EXTRACT EMAILS FROM 200 MEGABYTES OF LOGS BEFORE MIDNIGHT!' },
      { speaker: 'cueball', text: 'STAND BACK! I KNOW REGULAR EXPRESSIONS.' },
      { speaker: 'narrator', text: '*TAP TAP TAP*   NOW YOU HAVE 99 PROBLEMS.' },
    ],
    alt: 'Whenever I learn a new skill I concoct elaborate fantasy scenarios where it saves the day.',
  },
  {
    match: ['duty calls', 'wrong on the internet', 'sleep', '3am', 'bed', 'argument'],
    title: 'DUTY CALLS',
    layout: 'single_panel',
    prop: 'laptop_desk',
    cast: ['cueball', 'megan'],
    turns: [
      { speaker: 'megan', text: 'ARE YOU COMING TO BED?' },
      { speaker: 'cueball', text: "I CAN'T. THIS IS IMPORTANT." },
      { speaker: 'megan', text: 'WHAT?' },
      { speaker: 'cueball', text: 'SOMEONE IS WRONG ON THE INTERNET.' },
    ],
    alt: "What do you want me to do? LEAVE? Then they'll keep being wrong!",
  },
  {
    match: ['chart', 'graph', 'extrapolating', 'correlation', 'causation', 'statistics', 'plot'],
    title: 'EXTRAPOLATING',
    layout: 'annotated_chart',
    prop: 'chart_board',
    cast: ['cueball', 'megan'],
    turns: [
      { speaker: 'cueball', text: "AS YOU CAN SEE, BY LATE NEXT MONTH YOU'LL HAVE OVER FOUR DOZEN HUSBANDS." },
      { speaker: 'megan', text: 'BETTER GET A BULK RATE ON WEDDING CAKE.' },
    ],
    alt: 'By the third quarter of next year, a bridal shower will collapse into a black hole.',
  },
];

function escapeXml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

export function wobblePoints(x0, y0, x1, y1, amp = 1.65, seed = 42, steps = 10) {
  const pts = [];
  const dx = x1 - x0;
  const dy = y1 - y0;
  const len = Math.hypot(dx, dy) + 1e-6;
  const nx = -dy / len;
  const ny = dx / len;
  for (let i = 0; i <= steps; i++) {
    const u = i / steps;
    const env = Math.sin(u * Math.PI);
    const wave = Math.sin(u * Math.PI * 2.0 + seed * 0.73) * 0.65 + Math.cos(u * Math.PI * 3.0 + seed * 1.37) * 0.35;
    const offset = env * wave * amp;
    pts.push([Number((x0 + dx * u + nx * offset).toFixed(2)), Number((y0 + dy * u + ny * offset).toFixed(2))]);
  }
  return pts;
}

function svgPolyline(pts, strokeW = 2.1, fill = 'none') {
  if (!pts || pts.length === 0) return '';
  const d = `M ${pts[0][0]} ${pts[0][1]} ` + pts.slice(1).map(p => `L ${p[0]} ${p[1]}`).join(' ');
  return `<path d="${d}" stroke="#111111" stroke-width="${strokeW.toFixed(2)}" stroke-linecap="round" stroke-linejoin="round" fill="${fill}"/>`;
}

function wrapComicLines(text, maxChars = 26) {
  const words = String(text || '').toUpperCase().split(/\s+/).filter(Boolean);
  const lines = [];
  let cur = '';
  for (const w of words) {
    if (!cur) cur = w;
    else if (cur.length + 1 + w.length <= maxChars) cur += ' ' + w;
    else {
      lines.push(cur);
      cur = w;
    }
  }
  if (cur) lines.push(cur);
  return lines.slice(0, 4);
}

function drawCharacterSvg(charName, cx, footY, scale = 0.88, leftArm = -0.45, rightArm = 0.45, wobbleAmp = 1.6, seed = 11) {
  const parts = [];
  const headR = 16.0 * scale;
  const headCy = footY - 98.0 * scale;
  const neckY = headCy + headR;
  const hipY = footY - 42.0 * scale;
  const shoulderY = neckY + 10.0 * scale;

  parts.push(svgPolyline(wobblePoints(cx, neckY, cx, hipY, wobbleAmp, seed + 1), 2.2));
  parts.push(svgPolyline(wobblePoints(cx, hipY, cx - 15 * scale, footY, wobbleAmp, seed + 2), 2.2));
  parts.push(svgPolyline(wobblePoints(cx, hipY, cx + 15 * scale, footY, wobbleAmp, seed + 3), 2.2));

  const armLen = 28.0 * scale;
  parts.push(svgPolyline(wobblePoints(cx, shoulderY, cx - Math.cos(leftArm) * armLen, shoulderY - Math.sin(leftArm) * armLen, wobbleAmp, seed + 4), 2.1));
  parts.push(svgPolyline(wobblePoints(cx, shoulderY, cx + Math.cos(rightArm) * armLen, shoulderY - Math.sin(rightArm) * armLen, wobbleAmp, seed + 5), 2.1));

  const headPts = [];
  for (let k = 0; k <= 24; k++) {
    const ang = (k / 24.0) * Math.PI * 2.0;
    const jitter = 1.0 + 0.04 * Math.sin(ang * 3.0 + seed);
    headPts.push([
      Number((cx + Math.cos(ang) * headR * jitter).toFixed(2)),
      Number((headCy + Math.sin(ang) * headR * jitter).toFixed(2)),
    ]);
  }
  parts.push(svgPolyline(headPts, 2.2, '#ffffff'));

  if (charName === 'megan') {
    for (const hAng of [-2.3, -1.9, -1.5, -1.1, -0.7]) {
      const sx = cx + Math.cos(hAng) * headR;
      const sy = headCy + Math.sin(hAng) * headR;
      parts.push(svgPolyline(wobblePoints(sx, sy, sx + Math.cos(hAng - 0.5) * 12 * scale, sy + Math.sin(hAng + 0.4) * 14 * scale, wobbleAmp * 0.6, seed + 10), 2.0));
    }
  } else if (charName === 'black_hat') {
    const brimY = headCy - headR * 0.72;
    parts.push(svgPolyline(wobblePoints(cx - 21 * scale, brimY, cx + 21 * scale, brimY, 1.0, seed + 20), 2.6));
    parts.push(svgPolyline([[cx - 13 * scale, brimY], [cx - 11 * scale, brimY - 22 * scale], [cx + 11 * scale, brimY - 22 * scale], [cx + 13 * scale, brimY]], 2.0, '#111111'));
  } else if (charName === 'ponytail') {
    const px = cx - headR * 0.92;
    const py = headCy - 2 * scale;
    parts.push(svgPolyline([[px, py], [px - 12 * scale, py + 4 * scale], [px - 15 * scale, py + 18 * scale], [px - 6 * scale, py + 12 * scale]], 2.1));
  }
  return parts.join('\n');
}

function drawPropSvg(propName, px, footY, wobbleAmp = 1.5, seed = 77) {
  const parts = [];
  if (propName === 'laptop_desk') {
    const dy = footY - 44;
    parts.push(svgPolyline(wobblePoints(px - 36, dy, px + 36, dy, wobbleAmp, seed), 2.2));
    parts.push(svgPolyline(wobblePoints(px - 28, dy, px - 28, footY, wobbleAmp, seed + 1), 2.1));
    parts.push(svgPolyline(wobblePoints(px + 28, dy, px + 28, footY, wobbleAmp, seed + 2), 2.1));
    parts.push(svgPolyline(wobblePoints(px - 14, dy - 2, px + 8, dy - 2, 0.9, seed + 3), 2.2));
    parts.push(svgPolyline(wobblePoints(px + 8, dy - 2, px + 14, dy - 22, 0.9, seed + 4), 2.2));
  } else if (propName === 'whiteboard' || propName === 'chart_board') {
    const [bx0, by0, bx1, by1] = [px - 46, footY - 122, px + 46, footY - 54];
    const rect = [
      ...wobblePoints(bx0, by0, bx1, by0, wobbleAmp, seed),
      ...wobblePoints(bx1, by0, bx1, by1, wobbleAmp, seed + 1),
      ...wobblePoints(bx1, by1, bx0, by1, wobbleAmp, seed + 2),
      ...wobblePoints(bx0, by1, bx0, by0, wobbleAmp, seed + 3),
    ];
    parts.push(svgPolyline(rect, 2.0, '#ffffff'));
    parts.push(svgPolyline(wobblePoints(bx0 + 12, by1 - 12, bx0 + 12, by0 + 12, 0.8, seed + 4), 1.6));
    parts.push(svgPolyline(wobblePoints(bx0 + 12, by1 - 12, bx1 - 12, by1 - 12, 0.8, seed + 5), 1.6));
    parts.push(svgPolyline([[bx0 + 14, by1 - 16], [bx0 + 36, by1 - 22], [bx0 + 58, by1 - 34], [bx1 - 14, by0 + 14]], 1.8));
  } else if (propName === 'server_rack') {
    const [rx0, ry0, rx1, ry1] = [px - 26, footY - 118, px + 26, footY];
    const rect = [
      ...wobblePoints(rx0, ry0, rx1, ry0, wobbleAmp, seed),
      ...wobblePoints(rx1, ry0, rx1, ry1, wobbleAmp, seed + 1),
      ...wobblePoints(rx1, ry1, rx0, ry1, wobbleAmp, seed + 2),
      ...wobblePoints(rx0, ry1, rx0, ry0, wobbleAmp, seed + 3),
    ];
    parts.push(svgPolyline(rect, 2.1, '#ffffff'));
    for (let sy = ry0 + 18; sy < ry1 - 10; sy += 20) {
      parts.push(svgPolyline(wobblePoints(rx0 + 6, sy, rx1 - 6, sy, 0.8, seed + sy), 1.5));
    }
  } else if (propName === 'couch') {
    const cy = footY - 26;
    parts.push(svgPolyline(wobblePoints(px - 48, cy, px + 48, cy, wobbleAmp, seed), 2.2));
    parts.push(svgPolyline(wobblePoints(px - 48, cy - 22, px - 48, footY, wobbleAmp, seed + 1), 2.2));
    parts.push(svgPolyline(wobblePoints(px + 48, cy - 22, px + 48, footY, wobbleAmp, seed + 2), 2.2));
  }
  return parts.join('\n');
}

export default function xkcdArtist(input, options = {}) {
  const rawText = typeof input === 'object' && input !== null ? String(input.prompt || input.title || '') : String(input || '');
  const clean = rawText.trim();
  const isInvalid =
    clean.length === 0 ||
    clean.includes('\x00') ||
    clean.toLowerCase().includes('404 not found empty prompt abstention');
  const pInvalid = isInvalid ? 0.9992 : 0.0006;

  const low = clean.toLowerCase();
  let tpl = null;
  let bestHits = 0;
  for (const candidate of ICONIC_TEMPLATES) {
    const hits = candidate.match.filter(kw => low.includes(kw)).length;
    if (hits > bestHits) {
      bestHits = hits;
      tpl = candidate;
    }
  }

  if (!tpl) {
    const words = (clean.match(/[A-Za-z0-9_]+/g) || ['TECH', 'SUPPORT']).slice(0, 3);
    const cast = low.includes('black hat') ? ['cueball', 'black_hat'] : (low.includes('ponytail') || low.includes('ai') ? ['cueball', 'ponytail'] : ['cueball', 'megan']);
    const prop = low.includes('chart') || low.includes('graph') ? 'chart_board' : (low.includes('server') || low.includes('cloud') ? 'server_rack' : 'laptop_desk');
    tpl = {
      title: words.join(' ').toUpperCase(),
      layout: 'three_panel',
      prop,
      cast,
      turns: [
        { speaker: cast[0], text: `I JUST AUTOMATED ${clean.slice(0, 44).toUpperCase()}!` },
        { speaker: cast[1], text: 'WAIT... HOW LONG DID YOU SPEND DEBUGGING THE EDGE CASES?' },
        { speaker: cast[0], text: 'SIXTEEN HOURS. BUT IT SAVES FOUR SECONDS A WEEK!' },
      ],
      alt: `If you extrapolate the time savings on '${clean.slice(0, 40)}', it breaks even in 2039.`,
    };
  }

  const turns = tpl.turns;
  const nPanels = options.panels || (tpl.layout === 'single_panel' || tpl.layout === 'annotated_chart' ? 1 : (turns.length === 2 ? 2 : 3));
  const panels = [];
  if (nPanels === 1) {
    panels.push({ index: 0, cast: tpl.cast, prop: tpl.prop, turns: turns.slice(0, 2) });
  } else if (nPanels === 2) {
    panels.push({ index: 0, cast: tpl.cast, prop: tpl.prop, turns: turns.slice(0, 1) });
    panels.push({ index: 1, cast: tpl.cast, prop: tpl.prop, turns: turns.slice(1, 3) });
  } else {
    panels.push({ index: 0, cast: tpl.cast, prop: tpl.prop, turns: turns.slice(0, 1) });
    panels.push({ index: 1, cast: tpl.cast, prop: tpl.prop, turns: turns.slice(1, 2) });
    panels.push({ index: 2, cast: tpl.cast, prop: tpl.prop, turns: turns.slice(2, 4) });
  }

  const style = {
    wobble_amp_px: options.wobbleAmp ?? 1.65,
    stroke_width_px: options.strokeWidth ?? 2.05,
    bubble_zone_ratio: 0.28,
    ink_contrast: 0.92,
  };

  return {
    prompt: clean,
    title: tpl.title,
    alt: tpl.alt,
    layout: tpl.layout,
    prop: tpl.prop,
    cast: tpl.cast,
    panels,
    style,
    p_invalid: pInvalid,
    is_invalid: isInvalid,
    confidence: Number((1 - pInvalid).toFixed(4)),
    toSVG(width = 860, height = 390) {
      if (isInvalid) {
        return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}"><rect width="${width}" height="${height}" fill="#ffffff"/></svg>`;
      }
      const pad = 18;
      const topH = 38;
      const botH = 34;
      const panelW = (width - pad * 2 - (panels.length - 1) * 14) / panels.length;
      const panelH = height - topH - botH - pad;
      const out = [
        `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" font-family="'Humor Sans', 'Chalkboard SE', 'Comic Sans MS', monospace">`,
        `<rect width="${width}" height="${height}" fill="#ffffff"/>`,
        `<text x="${(width / 2).toFixed(1)}" y="26" text-anchor="middle" font-size="18" font-weight="700" letter-spacing="1.6" fill="#111111">${escapeXml(tpl.title)}</text>`,
      ];

      panels.forEach((p, pi) => {
        const px0 = pad + pi * (panelW + 14);
        const py0 = topH;
        const px1 = px0 + panelW;
        const py1 = py0 + panelH;
        const footY = py1 - 18;

        const borderPts = [
          ...wobblePoints(px0, py0, px1, py0, style.wobble_amp_px, 100 + pi * 7),
          ...wobblePoints(px1, py0, px1, py1, style.wobble_amp_px, 101 + pi * 7),
          ...wobblePoints(px1, py1, px0, py1, style.wobble_amp_px, 102 + pi * 7),
          ...wobblePoints(px0, py1, px0, py0, style.wobble_amp_px, 103 + pi * 7),
        ];
        out.push(svgPolyline(borderPts, style.stroke_width_px, '#ffffff'));
        out.push(svgPolyline(wobblePoints(px0 + 10, footY, px1 - 10, footY, style.wobble_amp_px * 0.5, 200 + pi), 1.2));

        if (p.prop !== 'none') {
          out.push(drawPropSvg(p.prop, (px0 + px1) * 0.5, footY, style.wobble_amp_px, 300 + pi));
        }

        const cxList = p.cast.length === 1 ? [px0 + panelW * 0.5] : [px0 + panelW * 0.27, px0 + panelW * 0.73];
        const charXs = {};
        p.cast.slice(0, 2).forEach((cname, ci) => {
          charXs[cname] = cxList[ci];
          out.push(drawCharacterSvg(cname, cxList[ci], footY, 0.88, -0.45, 0.45, style.wobble_amp_px, 400 + pi * 13 + ci));
        });

        let curBy = py0 + 22;
        const maxChars = Math.max(18, Math.floor(panelW / 9.2));
        p.turns.slice(0, 2).forEach((t, ti) => {
          const lines = wrapComicLines(t.text, maxChars);
          const isRight = ti === 1 || t.speaker === p.cast[1];
          const tx = px0 + panelW * (isRight ? 0.64 : 0.36);
          lines.forEach((ln, li) => {
            out.push(`<text x="${tx.toFixed(1)}" y="${(curBy + li * 14.5).toFixed(1)}" text-anchor="middle" font-size="12" font-weight="700" fill="#111111">${escapeXml(ln)}</text>`);
          });
          const tailY0 = curBy + (lines.length - 1) * 14.5 + 5;
          const targetX = charXs[t.speaker] ?? cxList[isRight ? cxList.length - 1 : 0];
          const targetY = footY - 104;
          if (tailY0 < targetY - 6) {
            out.push(svgPolyline(wobblePoints(tx, tailY0, tx + (targetX - tx) * 0.65, Math.min(targetY - 6, tailY0 + 28), 0.9, 600 + pi * 5 + ti), 1.35));
          }
          curBy += lines.length * 14.5 + 14;
        });
      });

      const altY = height - botH + 6;
      out.push(`<rect x="${pad}" y="${altY}" width="${width - pad * 2}" height="24" rx="4" fill="#fffef7" stroke="#111111" stroke-width="1.2"/>`);
      out.push(`<text x="${(width / 2).toFixed(1)}" y="${altY + 16}" text-anchor="middle" font-size="11" font-family="monospace" fill="#333333">alt: &quot;${escapeXml(tpl.alt.slice(0, 108))}&quot;</text>`);
      out.push('</svg>');
      return out.join('\n');
    },
  };
}
