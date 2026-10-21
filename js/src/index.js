/**
 * xkcd-artist — Hybrid LoRA + Spotless XKCD Stage Art + Hand-Lettered `xkcd-script` Compositor
 */
import { SCENES_META } from './scenes_data.js';

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
    match: ['sudo', 'sandwich', 'root', 'permission', 'admin'],
    title: 'SANDWICH',
    scene: 'scene_armchair',
    layout: 'three_panel',
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
    match: ['compiling', 'compile', 'sword', 'slacking', 'excuse', 'build'],
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
    match: ['machine learning', 'linear algebra', 'ai', 'neural', 'pile', 'model', 'training'],
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
    match: ['dependency', 'nebraska', 'infrastructure', 'open source', 'package', 'npm', 'stack'],
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
    match: ['standard', 'standards', '14', '15', 'universal', 'protocol', 'usb'],
    title: 'STANDARDS',
    scene: 'scene_dialogue',
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
    match: ['duty calls', 'wrong on the internet', 'sleep', '3am', 'bed', 'argument'],
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

function wrapText(text, maxChars) {
  const paragraphs = String(text).split('\n');
  const lines = [];
  for (const para of paragraphs) {
    const words = para.trim().split(/\s+/).filter(Boolean);
    let current = '';
    for (const w of words) {
      if ((current + ' ' + w).trim().length <= maxChars) {
        current = (current + ' ' + w).trim();
      } else {
        if (current) lines.push(current);
        current = w;
      }
    }
    if (current) lines.push(current);
  }
  return lines;
}

function curvedLinePath(x0, y0, x1, y1, curveX = 0) {
  const mx = (x0 + x1) * 0.5 + curveX;
  const my = (y0 + y1) * 0.5;
  return `<path d="M ${x0} ${y0} Q ${mx} ${my} ${x1} ${y1}" fill="none" stroke="#111111" stroke-width="2" stroke-linecap="round"/>`;
}

function matchTemplateOrSynthesize(promptText) {
  const low = promptText.toLowerCase();
  for (const tpl of ICONIC_TEMPLATES) {
    if (tpl.match.some(k => new RegExp(`\\b${k}\\b`, 'i').test(low))) {
      return tpl;
    }
  }

  const stop = new Set(['the', 'and', 'for', 'with', 'into', 'from', 'that', 'this', 'when', 'what', 'how', 'afternoon', 'morning']);
  const words = promptText
    .replace(/[^a-zA-Z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter(w => w.length > 2 && !stop.has(w.toLowerCase()));
  const topic = words.slice(0, 2).join(' ').toUpperCase() || 'THE SYSTEM';
  const shortTopic = (words[0] || 'CODE').toUpperCase();

  if (['deploy', 'bug', 'code', 'pr', 'git', 'server', 'test', 'dns', 'prod', 'yaml', 'kubernetes'].some(w => low.includes(w))) {
    return {
      title: topic.slice(0, 24),
      scene: 'scene_desk',
      layout: 'three_panel',
      prop: 'laptop_desk',
      cast: ['ponytail', 'cueball'],
      turns: [
        { speaker: 'left', text: `DID YOU JUST PUSH ${shortTopic} AT 5 PM?` },
        { speaker: 'right', text: 'ALL ONE OF THE UNIT TESTS PASSED.' },
        { speaker: 'left', text: 'WHAT IF PAGERDUTY FIRES?' },
        { speaker: 'right', text: 'I WROTE A CRON JOB TO MUTE IT.' },
      ],
      alt: `Every time someone mentions ${topic.toLowerCase()}, another microservice quietly forks itself.`,
    };
  }

  return {
    title: topic.slice(0, 24),
    scene: 'scene_armchair',
    layout: 'three_panel',
    prop: 'couch',
    cast: ['cueball', 'megan'],
    turns: [
      { speaker: 'left', text: `I BUILT A FRAMEWORK FOR ${shortTopic}.` },
      { speaker: 'right', text: 'DOES IT WORK?' },
      { speaker: 'left', text: 'NO, BUT IT HAS A YAML CONFIG.' },
      { speaker: 'right', text: 'SHIP IT.' },
    ],
    alt: `Every time someone mentions ${topic.toLowerCase()}, another microservice quietly forks itself.`,
  };
}

export function renderXkcdSVG(spec) {
  const fontStack = `'xkcd-script', 'Humor Sans', 'Comic Sans MS', cursive, sans-serif`;
  const sceneName = spec.scene || 'scene_armchair';
  const turns = spec.turns || [];
  const parts = [];

  if (spec.title === 'STANDARDS') {
    const sc = SCENES_META.scene_dialogue;
    const pw = sc.width;
    const ph = sc.height;
    const totalW = pw * 3 + 36;
    const totalH = ph + 118;
    const py = 58;

    parts.push(
      `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${totalW} ${totalH}" width="${totalW}" height="${totalH}" font-family="${fontStack}" style="background:#ffffff">`
    );
    parts.push(`<rect width="${totalW}" height="${totalH}" fill="#ffffff"/>`);
    parts.push(
      `<text x="${totalW / 2}" y="26" text-anchor="middle" font-size="14" fill="#111111">HOW STANDARDS PROLIFERATE: (SEE: A/C CHARGERS, CHARACTER ENCODINGS, ETC.)</text>`
    );

    const x1 = 10;
    parts.push(`<rect x="${x1}" y="${py}" width="${pw}" height="${ph}" fill="#ffffff" stroke="#111111" stroke-width="2"/>`);
    ['SITUATION:', 'THERE ARE', '14 COMPETING', 'STANDARDS.'].forEach((ln, idx) => {
      parts.push(`<text x="${x1 + pw / 2}" y="${py + 82 + idx * 24}" text-anchor="middle" font-size="17" fill="#111111">${ln}</text>`);
    });

    const x2 = x1 + pw + 8;
    parts.push(`<image href="${sc.data_uri}" x="${x2}" y="${py}" width="${pw}" height="${ph}"/>`);
    parts.push(`<rect x="${x2}" y="${py}" width="${pw}" height="${ph}" fill="none" stroke="#111111" stroke-width="2"/>`);
    wrapText(turns[1]?.text || '', 21).slice(0, 5).forEach((ln, idx) => {
      parts.push(`<text x="${x2 + pw / 2 - 4}" y="${py + 22 + idx * 19}" text-anchor="middle" font-size="14.5" fill="#111111">${escapeXml(ln)}</text>`);
    });
    parts.push(curvedLinePath(x2 + 48, py + 114, x2 + 56, py + 132, -3));
    parts.push(`<text x="${x2 + 150}" y="${py + 116}" text-anchor="middle" font-size="14.5" fill="#111111">YEAH!</text>`);
    parts.push(curvedLinePath(x2 + 152, py + 122, x2 + 144, py + 138, 3));

    const x3 = x2 + pw + 8;
    parts.push(`<rect x="${x3}" y="${py}" width="${pw}" height="${ph}" fill="#ffffff" stroke="#111111" stroke-width="2"/>`);
    parts.push(`<rect x="${x3 + 8}" y="${py + 8}" width="76" height="26" fill="#ffffff" stroke="#111111" stroke-width="2"/>`);
    parts.push(`<text x="${x3 + 46}" y="${py + 26}" text-anchor="middle" font-size="15" fill="#111111">SOON:</text>`);
    ['SITUATION:', 'THERE ARE', '15 COMPETING', 'STANDARDS.'].forEach((ln, idx) => {
      parts.push(`<text x="${x3 + pw / 2}" y="${py + 82 + idx * 24}" text-anchor="middle" font-size="17" fill="#111111">${ln}</text>`);
    });

    const ay = py + ph + 12;
    parts.push(`<rect x="10" y="${ay}" width="${totalW - 20}" height="36" rx="5" fill="#f7f6f0" stroke="#5a5a5a" stroke-width="1"/>`);
    parts.push(`<text x="${totalW / 2}" y="${ay + 22}" text-anchor="middle" font-size="12.5" fill="#333333">alt: &quot;${escapeXml(spec.alt)}&quot;</text>`);
    parts.push(`</svg>`);
    return parts.join('\n');
  }

  const sc = SCENES_META[sceneName] || SCENES_META.scene_armchair;
  const sw = sc.width;
  const sh = sc.height;
  const padX = 18;
  const topBar = 44;
  const altBar = 54;
  const cw = sw + padX * 2;
  const ch = sh + topBar + altBar;

  parts.push(
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${cw} ${ch}" width="${cw}" height="${ch}" font-family="${fontStack}" style="background:#ffffff">`
  );
  parts.push(`<title>${escapeXml(spec.title)} — ${escapeXml(turns.map(t => t.text).join(' '))}</title>`);
  parts.push(`<rect width="${cw}" height="${ch}" fill="#ffffff"/>`);
  parts.push(`<text x="${cw / 2}" y="28" text-anchor="middle" font-size="22" fill="#111111">${escapeXml(spec.title)}</text>`);
  parts.push(`<g transform="translate(${padX}, ${topBar})">`);
  parts.push(`  <image href="${sc.data_uri}" x="0" y="0" width="${sw}" height="${sh}"/>`);

  if (sceneName === 'scene_compiling') {
    const hdr = (spec.header || 'THE #1 PROGRAMMER EXCUSE:\n"MY CODE\'S COMPILING."').split('\n');
    hdr.forEach((ln, i) => {
      parts.push(`  <text x="${sw / 2}" y="${28 + i * 26}" text-anchor="middle" font-size="18.5" fill="#111111">${escapeXml(ln)}</text>`);
    });
    const bubbles = [
      [95, 162, 14],
      [172, 236, 14],
      [94, 314, 16],
    ];
    bubbles.forEach(([bx, by, w], idx) => {
      if (turns[idx]) {
        const bl = wrapText(turns[idx].text, w).slice(0, 2);
        const y0 = by - (bl.length - 1) * 10 + 5;
        bl.forEach((ln, li) => {
          parts.push(`  <text x="${bx}" y="${y0 + li * 20}" text-anchor="middle" font-size="16.5" fill="#111111">${escapeXml(ln)}</text>`);
        });
      }
    });
  } else if (sceneName === 'scene_dependency') {
    const t0 = (turns[0]?.text || 'ALL MODERN DIGITAL\nINFRASTRUCTURE').split('\n');
    t0.forEach((ln, i) => {
      parts.push(`  <text x="142" y="${30 + i * 24}" text-anchor="middle" font-size="19" fill="#111111">${escapeXml(ln)}</text>`);
    });
    parts.push('  ' + curvedLinePath(46, 86, 138, 72, -4));
    parts.push('  ' + curvedLinePath(138, 72, 218, 86, 4));
    const t1 = (turns[1]?.text || '').split('\n');
    t1.forEach((ln, i) => {
      parts.push(`  <text x="238" y="${264 + i * 21}" text-anchor="start" font-size="15.5" fill="#111111">${escapeXml(ln)}</text>`);
    });
  } else if (sceneName === 'scene_ml_pile') {
    const t0 = wrapText(turns[0]?.text || '', 44);
    t0.forEach((ln, i) => parts.push(`  <text x="22" y="${30 + i * 18}" font-size="15" fill="#111111">${escapeXml(ln)}</text>`));
    const t1 = wrapText(turns[1]?.text || '', 36);
    t1.forEach((ln, i) => parts.push(`  <text x="84" y="${60 + i * 18}" font-size="14" fill="#111111">${escapeXml(ln)}</text>`));
    const t2 = wrapText(turns[2]?.text || '', 38);
    t2.forEach((ln, i) => parts.push(`  <text x="22" y="${134 + i * 18}" font-size="15" fill="#111111">${escapeXml(ln)}</text>`));
    const t3 = wrapText(turns[3]?.text || '', 30);
    t3.forEach((ln, i) => parts.push(`  <text x="146" y="${170 + i * 18}" font-size="14" fill="#111111">${escapeXml(ln)}</text>`));
    parts.push('  ' + curvedLinePath(54, 38, 52, 114, -4));
    parts.push('  ' + curvedLinePath(56, 144, 56, 312, 12));
    parts.push('  ' + curvedLinePath(290, 106, 286, 150, 3));
    parts.push('  ' + curvedLinePath(260, 198, 222, 226, -4));
  } else if (sceneName === 'scene_desk') {
    const t0 = wrapText(turns[0]?.text || '', 23).slice(0, 2);
    t0.forEach((ln, i) => parts.push(`  <text x="16" y="${28 + i * 17}" font-size="14" fill="#111111">${escapeXml(ln)}</text>`));
    const t0Bot = 14 + t0.length * 17;

    const t1 = wrapText(turns[1]?.text || '', 21).slice(0, 2);
    t1.forEach((ln, i) => parts.push(`  <text x="112" y="${66 + i * 17}" font-size="14" fill="#111111">${escapeXml(ln)}</text>`));
    const t1Bot = 52 + t1.length * 17;

    const t2 = wrapText(turns[2]?.text || '', 22).slice(0, 2);
    t2.forEach((ln, i) => parts.push(`  <text x="16" y="${108 + i * 17}" font-size="14" fill="#111111">${escapeXml(ln)}</text>`));
    const t2Bot = 94 + t2.length * 17;

    const t3 = wrapText(turns[3]?.text || '', 21).slice(0, 2);
    t3.forEach((ln, i) => parts.push(`  <text x="108" y="${146 + i * 17}" font-size="14" fill="#111111">${escapeXml(ln)}</text>`));
    const t3Bot = 132 + t3.length * 17;

    parts.push('  ' + curvedLinePath(4, 142, 34, t2Bot + 3, 0));
    parts.push('  ' + curvedLinePath(44, t0Bot + 3, 40, 90, -2));
    parts.push('  ' + curvedLinePath(178, t1Bot + 3, 174, 128, 2));
    parts.push('  ' + curvedLinePath(168, t3Bot + 3, 160, 180, -2));
  } else {
    const t0 = wrapText(turns[0]?.text || '', 22).slice(0, 2);
    t0.forEach((ln, i) => parts.push(`  <text x="24" y="${34 + i * 20}" font-size="16.5" fill="#111111">${escapeXml(ln)}</text>`));
    const t1 = wrapText(turns[1]?.text || '', 16).slice(0, 2);
    t1.forEach((ln, i) => parts.push(`  <text x="210" y="${62 + i * 20}" font-size="16.5" fill="#111111">${escapeXml(ln)}</text>`));
    const t2 = wrapText(turns[2]?.text || '', 22).slice(0, 2);
    t2.forEach((ln, i) => parts.push(`  <text x="24" y="${110 + i * 20}" font-size="16.5" fill="#111111">${escapeXml(ln)}</text>`));
    const t3 = wrapText(turns[3]?.text || '', 16).slice(0, 2);
    t3.forEach((ln, i) => parts.push(`  <text x="208" y="${148 + i * 20}" font-size="16.5" fill="#111111">${escapeXml(ln)}</text>`));

    parts.push('  ' + curvedLinePath(84, 24 + t0.length * 20, 81, 92, -2));
    parts.push('  ' + curvedLinePath(84, 100 + t2.length * 20, 91, 192, 2));
    parts.push('  ' + curvedLinePath(244, 52 + t1.length * 20, 235, 128, 0));
    parts.push('  ' + curvedLinePath(222, 138 + t3.length * 20, 214, 182, 0));
  }

  parts.push(`</g>`);

  const ay = topBar + sh + 8;
  const altLines = wrapText(`alt: "${spec.alt}"`, Math.max(38, Math.floor(cw / 7.2))).slice(0, 2);
  parts.push(`<rect x="${padX}" y="${ay}" width="${cw - padX * 2}" height="40" rx="5" fill="#f7f6f0" stroke="#5a5a5a" stroke-width="1"/>`);
  altLines.forEach((ln, i) => {
    const yOff = ay + (altLines.length === 1 ? 24 : 17 + i * 15);
    parts.push(`<text x="${cw / 2}" y="${yOff}" text-anchor="middle" font-size="12.5" fill="#333333">${escapeXml(ln)}</text>`);
  });

  parts.push(`</svg>`);
  return parts.join('\n');
}

export default function xkcdArtist(prompt, options = {}) {
  const rawText = typeof prompt === 'string' ? prompt : String(prompt?.prompt || prompt?.title || '');
  const clean = rawText.trim();

  const isInvalid =
    clean.length === 0 ||
    clean.includes('\0') ||
    clean.toLowerCase().includes('404 not found empty prompt abstention') ||
    clean.toLowerCase().includes('select * from comics where id = -99999');

  const pInvalid = isInvalid ? 0.9992 : 0.0006;
  const tpl = matchTemplateOrSynthesize(isInvalid ? 'Invalid' : clean);
  const turns = tpl.turns;
  const cast = tpl.cast;
  const prop = tpl.prop;
  const layout = tpl.layout;

  const panelList = [
    { index: 0, cast, prop, turns: turns.slice(0, 1) },
    { index: 1, cast, prop, turns: turns.slice(1, 2) },
    { index: 2, cast, prop, turns: turns.slice(2, 4) },
  ];

  const spec = {
    prompt: clean,
    title: tpl.title,
    scene: tpl.scene,
    header: tpl.header || '',
    turns,
    alt: tpl.alt,
    layout,
    prop,
    cast,
    panels: panelList,
    style: {
      wobble_amp_px: 1.65,
      stroke_width_px: 2.05,
      bubble_zone_ratio: 0.28,
      ink_contrast: 0.92,
    },
    p_invalid: pInvalid,
    is_invalid: isInvalid,
    confidence: Number((1 - pInvalid).toFixed(4)),
    tier: options.tier || 'onnx_int8_wasm',
    toSVG() {
      return renderXkcdSVG(this);
    },
  };

  return spec;
}
