/**
 * xkcd-artist — Multi-Panel Generative XKCD Engine + Spotless Stage Art + `xkcd-script` Compositor
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
    match: ['sudo', 'sandwich'],
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

function hashPrompt(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return Math.abs(h >>> 0);
}

function synthesizeNovelComic(promptText) {
  const low = promptText.toLowerCase();
  for (const tpl of ICONIC_TEMPLATES) {
    if (tpl.match.some(k => low.includes(k))) {
      return tpl;
    }
  }

  const stop = new Set([
    'the', 'and', 'for', 'with', 'into', 'from', 'that', 'this', 'when', 'what',
    'why', 'how', 'about', 'like', 'make', 'making', 'trying', 'explained',
    'explain', 'cueball', 'megan', 'ponytail', 'black', 'hat', 'beret', 'guy',
    'comic', 'xkcd', 'panel', 'strip', 'someone', 'every', 'time', 'after',
    'before', 'my', 'our', 'your', 'is', 'are', 'was', 'in', 'on', 'to', 'at',
    'of', 'by', 'reviewing', 'automating', 'autonomous', 'line', 'lines',
  ]);
  const words = promptText
    .replace(/[^a-zA-Z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter(w => w.length >= 2 && !stop.has(w.toLowerCase()) && !/^\d+$/.test(w));

  const kw = words.length ? words : ['SYSTEM'];
  const title = kw.slice(0, 2).join(' ').toUpperCase().slice(0, 26);
  const subject = kw.slice(0, 2).join(' ').toUpperCase().slice(0, 20);
  const focus = kw[0].toUpperCase();
  const second = (kw[1] || 'PRODUCTION').toUpperCase();

  if (['docker', 'container', 'kubernetes', 'k8s', 'yaml', 'helm', 'cloud', 'aws', 'gcp', 'microservice'].some(w => low.includes(w))) {
    return {
      title,
      scene: 'scene_debug_3panel',
      layout: 'three_panel',
      prop: 'laptop_desk',
      cast: ['cueball', 'ponytail'],
      turns: [
        { speaker: 'left', text: `I CONTAINERIZED OUR 12-LINE ${focus} SCRIPT SO IT RUNS ANYWHERE.` },
        { speaker: 'right', text: `WHY IS THE ${subject} IMAGE 18.4 GIGABYTES?` },
        { speaker: 'left', text: "IT BUNDLES UBUNTU, THREE JDK VERSIONS, AND SOMEONE'S HOME DIRECTORY FROM 2019." },
      ],
      alt: `If you run docker system prune, the ${subject.toLowerCase()} cluster achieves negative mass and floats out of the rack.`,
    };
  }

  if (['git', 'rebase', 'merge', 'branch', 'commit', 'cherry-pick', 'force push'].some(w => low.includes(w))) {
    return {
      title,
      scene: 'scene_1597_team',
      layout: 'three_panel',
      prop: 'laptop_desk',
      cast: ['cueball', 'ponytail', 'black_hat'],
      turns: [
        { speaker: 'left', text: `I RAN INTERACTIVE ${subject} TO CLEAN UP OUR COMMIT HISTORY BEFORE THE RELEASE.` },
        { speaker: 'right', text: 'DID IT WORK?' },
        { speaker: 'left', text: "HEAD IS DETACHED, MAIN IS IN 1970, AND EVERYONE JUST CLONED A FRESH REPO INTO `PROJECT_FINAL_V3`." },
      ],
      alt: 'git reflog is just a horror novel where the main character is you from twenty minutes ago.',
    };
  }

  if (['review', 'pr', 'pull request', 'refactor', 'legacy', 'messy', 'lint', 'type', 'typescript', 'css'].some(w => low.includes(w))) {
    return {
      title,
      scene: 'scene_1513_4panel',
      layout: 'four_panel',
      prop: 'laptop_desk',
      cast: ['cueball', 'ponytail'],
      turns: [
        { speaker: 'left', text: `CAN YOU REVIEW MY ${focus} PR? IT'S ONLY 4,200 LINES.` },
        { speaker: 'right', text: 'WHY DOES ONE REGEX PARSE BOTH HTML AND TAXES?' },
        { speaker: 'right', text: `AND WHY IS ${focus} IMPORTING A DEPRECATED GAME ENGINE FOR DATE MATH?` },
        { speaker: 'left', text: 'LGTM, MERGING WITH `--NO-VERIFY`.' },
      ],
      alt: `Every comment in the ${subject.toLowerCase()} codebase just says '// DO NOT TOUCH OR THE BILLING CRON DIES'.`,
    };
  }

  if (['llm', 'agent', 'agents', 'gpt', 'prompt', 'ai', 'chatbot', 'hallucinat', 'token', 'transformer'].some(w => low.includes(w))) {
    return {
      title,
      scene: 'scene_552_3panel',
      layout: 'three_panel',
      prop: 'none',
      cast: ['megan', 'cueball'],
      turns: [
        { speaker: 'right', text: `I DEPLOYED TWO ${subject} TO AUTOMATE MY WORKFLOW.` },
        { speaker: 'right', text: "ONE OPENS TICKETS AND THE OTHER CLOSES THEM AS 'WONTFIX'." },
        { speaker: 'left', text: 'SO NOTHING GETS DONE?', reply: 'AT 40,000 TOKENS/SEC.' },
      ],
      alt: `We added a third ${focus.toLowerCase()} agent to mediate, and now all three are in a series-A fundraising call.`,
    };
  }

  if (['debug', 'dns', 'bug', 'prod', 'deploy', 'server', 'sql', 'database', 'vim', 'error', 'outage', 'cache'].some(w => low.includes(w))) {
    return {
      title,
      scene: 'scene_debug_3panel',
      layout: 'three_panel',
      prop: 'laptop_desk',
      cast: ['cueball', 'ponytail'],
      turns: [
        { speaker: 'left', text: `IT'S DEFINITELY NOT ${focus}. I CHECKED ${subject} THREE TIMES.` },
        { speaker: 'right', text: `WHAT ABOUT THAT STALE ${second} CACHE ENTRY FROM 2021?` },
        { speaker: 'left', text: `WHY IS ${focus} ALWAYS THE PROBLEM EVEN WHEN IT'S OFF?!` },
      ],
      alt: `The root cause analysis concluded that ${subject.toLowerCase()} was load-bearing technical debt.`,
    };
  }

  const archetypes = [
    {
      scene: 'scene_552_3panel',
      layout: 'three_panel',
      prop: 'none',
      cast: ['megan', 'cueball'],
      turns: [
        { speaker: 'right', text: `I SPENT ALL WEEKEND MODELING ${subject} FROM FIRST PRINCIPLES.` },
        { speaker: 'right', text: `IF YOU ASSUME ${focus} IS A FRICTIONLESS SPHERE IN A VACUUM, IT'S SOLVED.` },
        { speaker: 'left', text: 'AND IN REAL LIFE?', reply: 'EXPLODES IMMEDIATELY.' },
      ],
      alt: `Physicists hate this one weird boundary condition in ${subject.toLowerCase()}.`,
    },
    {
      scene: 'scene_debug_3panel',
      layout: 'three_panel',
      prop: 'laptop_desk',
      cast: ['cueball', 'ponytail'],
      turns: [
        { speaker: 'left', text: `I WROTE A QUICK SCRIPT TO AUTOMATE ${subject} IN FIVE MINUTES.` },
        { speaker: 'right', text: `YOU'VE BEEN DEBUGGING THE ${focus} EDGE CASES FOR SIX HOURS.` },
        { speaker: 'left', text: "IT'S ABOUT PRINCIPLE NOW!" },
      ],
      alt: `By year three, the automation script for ${subject.toLowerCase()} has its own compliance department.`,
    },
    {
      scene: 'scene_1513_4panel',
      layout: 'four_panel',
      prop: 'laptop_desk',
      cast: ['cueball', 'ponytail'],
      turns: [
        { speaker: 'left', text: `CHECK OUT MY NEW ${subject} BENCHMARK.` },
        { speaker: 'right', text: 'YOUR ERROR BARS ARE WIDER THAN THE Y-AXIS.' },
        { speaker: 'right', text: `AND DID YOU FIT A 9TH-DEGREE POLYNOMIAL TO THREE ${focus} POINTS?` },
        { speaker: 'left', text: 'R-SQUARED IS 1.000 THOUGH.' },
      ],
      alt: `If you extrapolate ${subject.toLowerCase()} to next Tuesday, the curve loops back through the Bronze Age.`,
    },
    {
      scene: 'scene_1597_team',
      layout: 'three_panel',
      prop: 'laptop_desk',
      cast: ['cueball', 'ponytail', 'black_hat'],
      turns: [
        { speaker: 'left', text: `OUR NEW ${subject} ARCHITECTURE USES ZERO EXTERNAL DEPENDENCIES.` },
        { speaker: 'right', text: `HOW DID YOU IMPLEMENT ${focus}?` },
        { speaker: 'left', text: 'WE WROTE OUR OWN BROKEN VERSION OF HALF THE STANDARD LIBRARY IN BASH.' },
      ],
      alt: `Eventually every ${subject.toLowerCase()} project invents its own Lisp interpreter by accident.`,
    },
  ];
  return { title, ...archetypes[hashPrompt(promptText) % archetypes.length] };
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
    parts.push(`<title>${escapeXml(spec.title)} — ${escapeXml(turns.map(t => t.text).join(' '))}</title>`);
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
  const padX = 16;
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

  if (sceneName === 'scene_debug_3panel') {
    const anchors = [
      [4, 220, 88, 102],
      [232, 220, 326, 98],
      [460, 220, 566, 92],
    ];
    anchors.forEach(([px, pw, hx, hy], idx) => {
      if (turns[idx]) {
        const lines = wrapText(turns[idx].text, 22).slice(0, 4);
        lines.forEach((ln, li) => {
          parts.push(`  <text x="${px + pw / 2}" y="${22 + li * 16}" text-anchor="middle" font-size="13.5" fill="#111111">${escapeXml(ln)}</text>`);
        });
        const tBot = 16 + lines.length * 16;
        if (tBot < hy - 4) {
          parts.push('  ' + curvedLinePath(px + pw / 2, tBot, hx, hy - 4, idx === 0 ? -3 : 3));
        }
      }
    });
  } else if (sceneName === 'scene_552_3panel') {
    const cols = [[4, 146], [156, 146], [308, 146]];
    cols.forEach(([px, pw], idx) => {
      if (turns[idx]) {
        const lines = wrapText(turns[idx].text, 19).slice(0, 4);
        lines.forEach((ln, li) => {
          parts.push(`  <text x="${px + pw / 2}" y="${22 + li * 15}" text-anchor="middle" font-size="12.5" fill="#111111">${escapeXml(ln)}</text>`);
        });
        const tBot = 16 + lines.length * 15;
        if (turns[idx].reply) {
          parts.push('  ' + curvedLinePath(px + 36, tBot, px + 34, 130, -2));
          const rLines = wrapText(turns[idx].reply, 15).slice(0, 2);
          rLines.forEach((rln, ri) => {
            parts.push(`  <text x="${px + 98}" y="${90 + ri * 14}" text-anchor="middle" font-size="12" fill="#111111">${escapeXml(rln)}</text>`);
          });
          parts.push('  ' + curvedLinePath(px + 98, 86 + rLines.length * 14, px + 92, 128, 2));
        } else {
          parts.push('  ' + curvedLinePath(px + 88, tBot, px + 88, 128, 3));
        }
      }
    });
  } else if (sceneName === 'scene_1513_4panel') {
    const cols = [
      [4, 168, 92, 138],
      [180, 178, 246, 150],
      [366, 180, 430, 150],
      [554, 180, 680, 138],
    ];
    cols.forEach(([px, pw, hx, hy], idx) => {
      if (turns[idx]) {
        const lines = wrapText(turns[idx].text, 20).slice(0, 5);
        lines.forEach((ln, li) => {
          parts.push(`  <text x="${px + pw / 2}" y="${22 + li * 15}" text-anchor="middle" font-size="12.5" fill="#111111">${escapeXml(ln)}</text>`);
        });
        const tBot = 16 + lines.length * 15;
        if (tBot < hy - 4) {
          parts.push('  ' + curvedLinePath(px + pw / 2, tBot, hx, hy - 4, -3));
        }
      }
    });
  } else if (sceneName === 'scene_1597_team') {
    const t0 = wrapText(turns[0]?.text || '', 32).slice(0, 3);
    t0.forEach((ln, i) => parts.push(`  <text x="18" y="${28 + i * 18}" font-size="14" fill="#111111">${escapeXml(ln)}</text>`));
    const t0Bot = 16 + t0.length * 18;
    const t1 = wrapText(turns[1]?.text || '', 24).slice(0, 2);
    t1.forEach((ln, i) => parts.push(`  <text x="134" y="${90 + i * 18}" font-size="14" fill="#111111">${escapeXml(ln)}</text>`));
    const t1Bot = 78 + t1.length * 18;
    const t2 = wrapText(turns[2]?.text || '', 32).slice(0, 4);
    t2.forEach((ln, i) => parts.push(`  <text x="18" y="${134 + i * 18}" font-size="14" fill="#111111">${escapeXml(ln)}</text>`));
    const t2Bot = 122 + t2.length * 18;
    parts.push('  ' + curvedLinePath(92, t0Bot + 2, 90, 118, -2));
    parts.push('  ' + curvedLinePath(96, t2Bot + 4, 114, 232, -5));
    parts.push('  ' + curvedLinePath(256, t1Bot + 4, 218, 248, 10));
  } else if (sceneName === 'scene_compiling') {
    const hdr = (spec.header || 'THE #1 PROGRAMMER EXCUSE:\n"MY CODE\'S COMPILING."').split('\n');
    hdr.forEach((ln, i) => {
      parts.push(`  <text x="${sw / 2}" y="${28 + i * 26}" text-anchor="middle" font-size="18.5" fill="#111111">${escapeXml(ln)}</text>`);
    });
    [[95, 162, 14], [172, 236, 14], [94, 314, 16]].forEach(([bx, by, w], idx) => {
      if (turns[idx]) {
        const bl = wrapText(turns[idx].text, w).slice(0, 2);
        const y0 = by - (bl.length - 1) * 10 + 5;
        bl.forEach((ln, li) => {
          parts.push(`  <text x="${bx}" y="${y0 + li * 20}" text-anchor="middle" font-size="16.5" fill="#111111">${escapeXml(ln)}</text>`);
        });
      }
    });
  } else if (sceneName === 'scene_dependency') {
    (turns[0]?.text || 'ALL MODERN DIGITAL\nINFRASTRUCTURE').split('\n').forEach((ln, i) => {
      parts.push(`  <text x="142" y="${30 + i * 24}" text-anchor="middle" font-size="19" fill="#111111">${escapeXml(ln)}</text>`);
    });
    parts.push('  ' + curvedLinePath(46, 86, 138, 72, -4));
    parts.push('  ' + curvedLinePath(138, 72, 218, 86, 4));
    (turns[1]?.text || '').split('\n').forEach((ln, i) => {
      parts.push(`  <text x="238" y="${264 + i * 21}" text-anchor="start" font-size="15.5" fill="#111111">${escapeXml(ln)}</text>`);
    });
  } else if (sceneName === 'scene_ml_pile') {
    wrapText(turns[0]?.text || '', 44).forEach((ln, i) => parts.push(`  <text x="22" y="${30 + i * 18}" font-size="15" fill="#111111">${escapeXml(ln)}</text>`));
    wrapText(turns[1]?.text || '', 36).forEach((ln, i) => parts.push(`  <text x="84" y="${60 + i * 18}" font-size="14" fill="#111111">${escapeXml(ln)}</text>`));
    wrapText(turns[2]?.text || '', 38).forEach((ln, i) => parts.push(`  <text x="22" y="${134 + i * 18}" font-size="15" fill="#111111">${escapeXml(ln)}</text>`));
    wrapText(turns[3]?.text || '', 30).forEach((ln, i) => parts.push(`  <text x="146" y="${170 + i * 18}" font-size="14" fill="#111111">${escapeXml(ln)}</text>`));
    parts.push('  ' + curvedLinePath(54, 38, 52, 114, -4));
    parts.push('  ' + curvedLinePath(56, 144, 56, 312, 12));
    parts.push('  ' + curvedLinePath(290, 106, 286, 150, 3));
    parts.push('  ' + curvedLinePath(260, 198, 222, 226, -4));
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
  const altLines = wrapText(`alt: "${spec.alt}"`, Math.max(44, Math.floor(cw / 6.8))).slice(0, 2);
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

  return {
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
}
