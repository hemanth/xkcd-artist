/**
 * Compositional XKCD Humor, Clause-Aware Semantic Parser & Rhetorical Dialogue Engine
 *
 * Replaces single-keyword slot-stuffing with:
 * 1. Strict word-boundary multi-token domain routing (zero substring false-positives).
 * 2. Clause-aware syntactic extraction (`lhs vs rhs`, infinitive/gerund goals, noun phrases,
 *    secondary context clauses, singular/plural agreement, and article phonology).
 * 3. 120 distinct deadpan Randall-Munroe-style comedic blueprints across 15 domains with
 *    sub-intent scoring + deterministic FNV-1a diversity routing.
 */

export function hashStr(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return Math.abs(h >>> 0);
}

const STOP_WORDS = new Set([
  'the','a','an','and','or','but','in','on','at','to','for','of','with','by','from','into',
  'onto','over','under','between','after','before','during','while','without','through','about',
  'when','where','why','how','what','who','which','that','this','these','those','my','our',
  'your','its','their','his','her','is','are','was','were','be','been','being','have','has',
  'had','do','does','did','will','would','could','should','can','may','might','must','shall',
  'it','i','we','you','they','me','us','them','so','as','if','than','too','very','just',
  'only','still','even','ever','never','always','every','some','any','no','not','all','both',
  'each','more','most','much','many','other','such','own','same','up','down','out','off',
  'again','back','there','here','now','then','also','well','get','got','getting','make',
  'making','made','take','taking','put','putting','use','using','try','trying','ask','asking',
  'write','writing','spend','spending','buy','buying','run','running','give','giving',
]);

const LEADING_GERUNDS = new Set([
  'training','trying','asking','writing','buying','getting','deploying','spending','making',
  'using','building','running','debugging','fixing','testing','explaining','teaching','learning',
  'reading','watching','waiting','painting','calculating','converting','comparing','choosing',
  'deciding','resolving','migrating','rotating','sharing','unmuting','ordering','boiling',
  'drinking','assembling','filing','canceling','throwing','pushing','adding','deleting',
  'centering','self-hosting','fine-tuning','dialing',
]);

const CHART_META_WORDS = new Set([
  'chart','graph','plot','curve','distribution','correlation','axis','axes','histogram',
  'flowchart','decision','tree','diagram','workflow','cheat','sheet','comic','xkcd','panel','strip',
]);

export function aAn(phrase) {
  const clean = String(phrase || '').trim();
  if (!clean) return 'a thing';
  const first = clean.split(/\s+/)[0].replace(/[^a-zA-Z0-9]/g, '');
  if (/^(8|11|18)(\b|\D)/.test(first)) return `an ${clean}`;
  if (/^(SQL|SSH|SSL|HTML|HTTP|HTTPS|RFC|IETF|MCP|LLM|API|FPS|SDK|SRE|ORM|NBA|IRS|NDA|HDMI|NVMe|SSD|HDD|AST)\b/.test(first)) {
    const acro = first.toUpperCase();
    const vowelSound = /^(SQL|SSH|SSL|HTML|HTTP|HTTPS|RFC|MCP|LLM|FPS|SDK|SRE|ORM|NBA|IRS|NDA|HDMI|NVME|SSD|HDD|AST)$/.test(acro);
    return `${vowelSound ? 'an' : 'a'} ${clean}`;
  }
  if (/^(uni|use|usa|usb|user|util|one|once|eu)/i.test(first)) return `a ${clean}`;
  if (/^(hour|honest|heir|herb)/i.test(first)) return `an ${clean}`;
  return /^[aeiou]/i.test(first) ? `an ${clean}` : `a ${clean}`;
}

export function singularize(word) {
  const w = String(word || '').trim();
  const low = w.toLowerCase();
  if (low.length <= 3) return w;
  const keepS = new Set([
    'dns','css','aws','k8s','https','serverless','kubernetes','series','species','physics',
    'mathematics','statistics','thermodynamics','mechanics','glasses','clothes','scissors',
    'pants','news','status','alias','canvas','bus','gas','lens','chaos','cosmos','hummus',
    'asparagus','sourdough','swiss','texas','mars','venus','postgres','redis',
  ]);
  if (keepS.has(low) || low.endsWith('ss') || low.endsWith('us') || low.endsWith('is')) return w;
  if (low.endsWith('ies') && low.length > 4) return w.slice(0, -3) + (w === w.toUpperCase() ? 'Y' : 'y');
  if (low.endsWith('ches') || low.endsWith('shes') || low.endsWith('xes') || low.endsWith('zes')) {
    return w.slice(0, -2);
  }
  if (low.endsWith('s')) return w.slice(0, -1);
  return w;
}

function isPluralWord(word) {
  const low = String(word || '').toLowerCase();
  if (['glasses','clothes','scissors','pants','people','children','mice','teeth','feet','data','criteria'].includes(low)) return true;
  if (['dns','css','aws','k8s','https','serverless','kubernetes','physics','thermodynamics','mechanics','mathematics','statistics','news','status','bus','gas','chaos','mars','venus','postgres','redis'].includes(low)) return false;
  if (low.endsWith('ss') || low.endsWith('us') || low.endsWith('is')) return false;
  return low.length > 3 && low.endsWith('s');
}

function stripLeadingArticles(s) {
  return s.replace(/^(the|a|an|my|our|your|this|that|these|those|some)\s+/i, '').trim();
}

export function parsePromptSemantics(rawPrompt) {
  const clean = String(rawPrompt || '').replace(/\s+/g, ' ').trim();
  const low = clean.toLowerCase();
  const tokens = low.match(/[a-z0-9'.+-]+/g) || [];
  const wordSet = new Set(tokens.map(t => t.replace(/^'+|'+$/g, '')));

  // 1. Detect X vs Y comparison
  let lhs = '';
  let rhs = '';
  const vsMatch =
    clean.match(/^(?:chart of|graph of|plot of|comparing|compare|difference between|choosing between)?\s*(.+?)\s+(?:vs\.?|versus)\s+(.+)$/i) ||
    clean.match(/^(?:choosing|difference|torn)\s+between\s+(.+?)\s+and\s+(.+)$/i);
  if (vsMatch) {
    lhs = stripLeadingArticles(vsMatch[1].replace(/^(chart|graph|plot|curve)\s+of\s+/i, '').trim());
    rhs = stripLeadingArticles(vsMatch[2].split(/\s+(?:during|when|while|before|after|in|on|at)\s+/i)[0].trim());
  }

  // 2. Extract infinitive goal ("to fold fitted bedsheets", "to center a div")
  let goalInf = '';
  const infMatch = clean.match(/\bto\s+([a-z][a-z0-9-]+\s+[^.,;!?]{3,42})/i);
  if (infMatch && !/^(the|a|an|my|our|this|be\s+honest)\b/i.test(infMatch[1])) {
    const cand = infMatch[1].split(/\s+(?:when|while|before|after|because|so\s+that)\b/i)[0].trim();
    if (/^[a-z]{3,12}\s+/i.test(cand) && !/^(day|date|work|bed|sleep|main|prod|production)$/i.test(cand)) {
      goalInf = cand;
    }
  }

  // 3. Split primary clause vs secondary context clause
  const prepSplit = clean.split(/\s+(?:before|after|during|while|without|across|until|because|when|under|inside|from|for|over|with|by)\s+/i);
  const primaryClause = prepSplit[0] || clean;
  const secondaryClause = prepSplit.slice(1).join(' ').trim();

  // 4. Content words (filtering chart meta words & leading filler gerunds)
  const rawWords = clean
    .split(/\s+/)
    .map(w => w.replace(/^[^a-zA-Z0-9$%.+-]+|[^a-zA-Z0-9%°+-]+$/g, ''))
    .filter(w => w.length >= 2);

  const contentWords = rawWords.filter(w => {
    const wl = w.toLowerCase();
    return !STOP_WORDS.has(wl) && !CHART_META_WORDS.has(wl);
  });

  const nonGerundWords = contentWords.filter((w, idx) => !(idx === 0 && LEADING_GERUNDS.has(w.toLowerCase())));
  const titlePool = nonGerundWords.length >= 1 ? nonGerundWords : contentWords.length ? contentWords : ['ENTROPY'];

  // Pick crisp 2-word XKCD title preferring concrete nouns over generic adjectives
  const genericAdj = new Set([
    'new','old','big','small','tiny','huge','simple','modern','legacy','custom','local','remote',
    'free','paid','fast','slow','clean','messy','weird','strange','mysterious','perfect','basic',
    'overpriced','artisanal','ambitious','unlabeled','undocumented','unattached','surprise',
  ]);
  const strongTitleWords = titlePool.filter(w => !genericAdj.has(w.toLowerCase()) && !/^\d+$/.test(w));
  const dedupPool = [];
  for (const w of (strongTitleWords.length >= 2 ? strongTitleWords : titlePool)) {
    if (!dedupPool.some(prev => prev.toLowerCase() === w.toLowerCase() || w.toLowerCase().startsWith(prev.toLowerCase()))) {
      dedupPool.push(w);
    }
  }
  const titleWords = (dedupPool.length ? dedupPool : titlePool).slice(0, 2);
  let rawTitle = titleWords.join(' ');
  if (lhs && rhs) {
    const cleanSide = s =>
      s
        .split(/\s+/)
        .map(w => w.replace(/^[^a-zA-Z0-9$%+-]+|[^a-zA-Z0-9%+-]+$/g, ''))
        .filter(w => w.length >= 2 && !STOP_WORDS.has(w.toLowerCase()) && !CHART_META_WORDS.has(w.toLowerCase()));
    const lw = cleanSide(lhs);
    const rw = cleanSide(rhs);
    const lShort = lw[lw.length - 1] || lhs.split(/\s+/)[0] || 'PLAN A';
    const rShort = rw[0] || rhs.split(/\s+/)[0] || 'PLAN B';
    rawTitle = `${lShort} VS ${rShort}`;
  }
  const titleParts = rawTitle.toUpperCase().split(/\s+/).filter(Boolean);
  while (titleParts.length > 1 && titleParts.join(' ').length > 26) {
    titleParts.pop();
  }
  const title = titleParts.join(' ').slice(0, 26) || 'EDGE CASE';

  // Build natural primary noun phrase (`topicNP`, e.g. "fitted bedsheets", "office printer", "espresso grind size")
  let topicNP = '';
  if (lhs && rhs) {
    topicNP = `${lhs} and ${rhs}`;
  } else {
    let cleanedPrimary = primaryClause
      .replace(/^(why|how|when|what\s+happens\s+when|that\s+moment\s+when)\s+/i, '')
      .replace(/^(chart|graph|plot|curve|distribution\s+curve|correlation\s+plot|flowchart|decision\s+tree|workflow\s+diagram|how\s+to\s+decide\s+cheat\s+sheet)\s+(of|for|between|on)?\s*/i, '')
      .trim();
    const firstTok = (cleanedPrimary.split(/\s+/)[0] || '').toLowerCase();
    if (LEADING_GERUNDS.has(firstTok)) {
      cleanedPrimary = cleanedPrimary.split(/\s+/).slice(1).join(' ');
    }
    cleanedPrimary = stripLeadingArticles(cleanedPrimary)
      .split(/\s+(?:smells|takes|taking|wants|wanting|needs|needing|looks|looking|goes|going|breaks|breaking|rejects|rejecting|dies|dying|runs|running|sits|sitting|joins|joining|asks|asking|ends|ending|defeats|defeating|reaches|reaching|requires|requiring|scores|scoring|shows|showing|wakes|waking|makes|making|invents|inventing|leaves|leaving|always|refuses|refusing|survives|surviving|tying|blowing|heating|snapping|wedged|stuck)\b/i)[0]
      .trim();
    const npWords = cleanedPrimary.split(/\s+/).filter(Boolean).slice(0, 4);
    while (npWords.length > 1 && STOP_WORDS.has(npWords[npWords.length - 1].toLowerCase())) {
      npWords.pop();
    }
    topicNP = npWords.length ? npWords.join(' ') : titleWords.join(' ').toLowerCase();
  }

  const npTokens = topicNP.split(/\s+/).filter(Boolean);
  const headWord = npTokens[npTokens.length - 1] || titleWords[titleWords.length - 1] || 'system';
  const isPlural = isPluralWord(headWord);
  const singularHead = singularize(headWord);
  const singularNP = [...npTokens.slice(0, -1), singularHead].join(' ');

  // Secondary concept phrase for multi-concept weaving
  const secWords = (secondaryClause || '')
    .split(/\s+/)
    .map(w => w.replace(/^[^a-zA-Z0-9$%+-]+|[^a-zA-Z0-9%°+-]+$/g, ''))
    .filter(w => w.length >= 3 && !STOP_WORDS.has(w.toLowerCase()));
  const contextTag = secWords.slice(0, 3).join(' ');

  // Number / unit cue if user supplied one (e.g. "4:55pm", "$4,800", "90 files", "3:14am")
  const numMatch = clean.match(/(\$?\d[\d,.]*(?:\s*(?:am|pm|ms|gb|mb|kb|tb|%|watts|hours|minutes|seconds|days|years|files|tabs|lines|miles|mm|cm))?)/i);
  const quantCue = numMatch ? numMatch[1].trim() : '';

  return {
    clean,
    low,
    tokens,
    wordSet,
    title,
    lhs,
    rhs,
    goalInf,
    topicNP,
    singularNP,
    headWord,
    singularHead,
    contextTag,
    secondaryClause: secondaryClause.slice(0, 44),
    quantCue,
    isPlural,
    itThey: isPlural ? 'they' : 'it',
    isAre: isPlural ? 'are' : 'is',
    doesDo: isPlural ? 'do' : 'does',
    themIt: isPlural ? 'them' : 'it',
    thisThese: isPlural ? 'these' : 'this',
  };
}

// Strict word/phrase boundary matcher (NEVER matches 'pr' inside 'printer' or 'ai' inside 'waiting'!)
function countMatches(low, wordSet, terms) {
  let score = 0;
  for (const [term, weight] of terms) {
    if (term.includes(' ')) {
      const escaped = term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      if (new RegExp(`\\b${escaped}\\b`, 'i').test(low)) score += weight;
    } else if (wordSet.has(term)) {
      score += weight;
    }
  }
  return score;
}

export function classifyPromptDomain(sem) {
  const { low, wordSet, lhs, rhs } = sem;

  if (countMatches(low, wordSet, [['flowchart', 6], ['flow chart', 6], ['decision tree', 6], ['cheat sheet', 5], ['workflow diagram', 6], ['how to decide', 5]]) >= 5) {
    return 'flowchart';
  }
  if (countMatches(low, wordSet, [['chart', 6], ['graph', 6], ['plot', 6], ['distribution curve', 6], ['correlation', 5], ['histogram', 6], ['bell curve', 6], ['x-axis', 6], ['y-axis', 6]]) >= 5) {
    return 'chart';
  }
  if (lhs && rhs) {
    return 'vs';
  }

  const domainLexicons = {
    ai: [
      ['ai', 4], ['llm', 5], ['llms', 5], ['gpt', 5], ['agent', 4], ['agents', 4], ['neural', 4],
      ['prompt', 4], ['prompts', 4], ['chatbot', 5], ['chatbots', 5], ['autocomplete', 4],
      ['language model', 6], ['fine-tuning', 5], ['fine-tune', 5], ['lora', 5], ['hallucinate', 5],
      ['hallucinating', 5], ['hallucinated', 5], ['tokens', 4], ['voice assistant', 5], ['coding assistant', 5],
    ],
    infra: [
      ['kubernetes', 5], ['k8s', 5], ['docker', 5], ['container', 4], ['containers', 4], ['terraform', 5],
      ['yaml', 5], ['aws', 5], ['gcp', 5], ['serverless', 5], ['nat gateway', 6], ['cloud bill', 6],
      ['microservice', 5], ['microservices', 5], ['nas', 5], ['self-hosting', 5], ['datacenter', 4], ['alpine', 4],
    ],
    code: [
      ['css', 5], ['div', 5], ['regex', 5], ['regular expression', 6], ['boolean', 5], ['variable', 4],
      ['rust', 5], ['borrow checker', 6], ['typescript', 5], ['compiler', 5], ['debugger', 5],
      ['dead code', 6], ['code comment', 6], ['print statements', 5], ['linked list', 5], ['safari', 4], ['pr', 4], ['pull request', 5],
    ],
    ops: [
      ['git', 5], ['rebase', 6], ['merge conflict', 6], ['commit', 4], ['force pushing', 6], ['force push', 6],
      ['dns', 6], ['ssl', 5], ['certificate', 4], ['pagerduty', 6], ['status page', 5], ['production', 4],
      ['prod', 4], ['outage', 5], ['rollback', 5], ['config', 4], ['friday', 3],
    ],
    science: [
      ['orbital', 5], ['mechanics', 4], ['relativistic', 5], ['thermodynamics', 6], ['linear algebra', 6],
      ['proof', 4], ['bayesian', 6], ['probability', 4], ['physicist', 6], ['frictionless', 6],
      ['exoplanet', 6], ['astronomers', 5], ['daylight saving', 5], ['clocks', 3], ['quantum', 5], ['mars', 4],
      ['cumulus', 6], ['clouds', 4], ['dew', 5],
    ],
    security: [
      ['password', 5], ['passwords', 5], ['two-factor', 6], ['2fa', 6], ['authentication', 5],
      ['cookie', 4], ['consent banner', 6], ['ssh', 6], ['phishing', 6], ['encrypting', 5],
      ['encryption', 5], ['security audit', 6], ['botnet', 6], ['smartfridge', 5], ['hackers', 4],
    ],
    work: [
      ['meeting', 5], ['meetings', 5], ['pre-meeting', 6], ['standup', 5], ['sync', 4], ['jira', 6],
      ['offline', 4], ['browser tabs', 4], ['spreadsheet', 5], ['vlookup', 6], ['vlookups', 6],
      ['calendar', 5], ['unmuting', 6], ['video call', 5], ['roadmap', 5], ['q4', 4], ['hr', 4], ['deadline', 3],
    ],
    food: [
      ['avocado', 6], ['kale', 6], ['meal-prep', 5], ['sourdough', 6], ['pasta', 6], ['boiling', 4],
      ['spicy', 5], ['thai', 5], ['waffle', 6], ['decaf', 6], ['espresso', 6], ['coffee', 5],
      ['kitchen drawer', 6], ['potato masher', 6], ['soup', 4], ['recipe', 4], ['recipes', 4],
    ],
    pets: [
      ['cat', 6], ['cats', 6], ['dog', 6], ['dogs', 6], ['golden retriever', 6], ['houseplant', 6],
      ['weeds', 4], ['crow', 6], ['zoomies', 6], ['seagull', 6], ['mosquito', 6], ['squirrel', 6],
      ['bird feeder', 6], ['velvet bed', 4], ['pet', 4],
    ],
    life: [
      ['furniture', 5], ['flat-pack', 6], ['swedish', 4], ['dowels', 5], ['taxes', 6], ['dentist', 6],
      ['gym membership', 6], ['gym', 4], ['shampoo', 6], ['conditioner', 6], ['parallel parking', 6],
      ['laundry', 6], ['alarm clock', 6], ['ceiling', 4], ['faucet', 5], ['airport', 5], ['boarding', 5], ['spring water', 5],
    ],
    net_hw: [
      ['domain name', 6], ['.dev', 5], ['wikipedia', 6], ['smart tv', 6], ['hdmi', 6], ['forum', 5],
      ['usb-a', 6], ['usb-c', 6], ['unsubscribe', 6], ['captive portal', 6], ['wi-fi', 5], ['wifi', 5],
      ['autocorrect', 6], ['printer', 6], ['bluetooth', 5],
    ],
  };

  let bestDomain = 'open_world';
  let bestScore = 0;
  for (const [dom, terms] of Object.entries(domainLexicons)) {
    const sc = countMatches(low, wordSet, terms);
    if (sc > bestScore) {
      bestScore = sc;
      bestDomain = dom;
    }
  }
  return bestScore >= 4 ? bestDomain : 'open_world';
}

export const ORIGINAL_SYNTHESIS_BANK = [
  {
    match: ['smart toaster', 'toaster firmware', 'bread-browning', 'quantized neural network inside a smart toaster'],
    title: 'SMART TOASTER',
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
      { speaker: 'left', text: 'MY NEW TOASTER NEEDS A 400MB FIRMWARE UPDATE BEFORE IT WILL MAKE BREAKFAST.' },
      { speaker: 'right', text: 'MINE JUST HAS A DIAL NUMBERED 1 THROUGH 6.' },
      { speaker: 'left', text: 'OH COOL, MINUTES?', reply: 'NOBODY KNOWS. 2 IS WARM BREAD AND 4 IS A HOUSE FIRE.' },
    ],
    alt: "Settings 5 and 6 exist strictly for tungsten annealing and voiding your renter's insurance.",
  },
  {
    match: ['mcp calendar', 'write access to my calendar', 'recruiter emails'],
    title: 'EMAIL AGENTS',
    scene: 'kinematic',
    layout: 'four_panel',
    prop: 'laptop_desk',
    cast: ['ponytail', 'cueball'],
    poses: [
      ['pointing', 'chin_think'],
      ['conversing', 'pointing'],
      ['shrug', 'chin_think'],
      ['conversing', 'hands_on_head'],
    ],
    turns: [
      { speaker: 'left', text: 'I SET UP AN AI AGENT TO POLITELY DECLINE RECRUITER EMAILS.' },
      { speaker: 'right', text: 'DOES IT WORK?' },
      { speaker: 'left', text: 'IT ACCIDENTALLY COUNTER-OFFERED AND NEGOTIATED A 25% RAISE.' },
      { speaker: 'right', text: 'CONGRATS!', reply: "MY FIRST STANDUP IS AT 2 PM AND I DON'T KNOW WHAT WE SELL." },
    ],
    alt: 'Apparently I am now Regional VP of Ball Bearings for the Greater Duluth Metro Area.',
  },
  {
    match: ['kv cache', 'h100 vram rent'],
    title: 'CONTEXT WINDOW',
    scene: 'kinematic',
    layout: 'two_panel',
    prop: 'whiteboard',
    cast: ['megan', 'cueball'],
    poses: [
      ['pointing', 'chin_think'],
      ['arms_crossed', 'hands_on_head'],
    ],
    turns: [
      { speaker: 'left', text: 'KEEPING YOUR 2-MILLION-TOKEN CHAT OPEN ON H100S COSTS $14 AN HOUR.' },
      { speaker: 'right', text: 'SO MY UNREAD PDF ATTACHMENTS NOW PAY HIGHER RENT THAN MY FIRST APARTMENT.' },
    ],
    alt: 'Closing the browser tab is legally classified as an eviction in three cloud availability zones.',
  },
  {
    match: ['quantization curve', '0.5 bits per weight', 'quantization cliff'],
    title: 'QUANTIZATION CLIFF',
    scene: 'kinematic',
    layout: 'annotated_chart',
    prop: 'chart_board',
    cast: ['cueball'],
    chart_meta: {
      curve_style: 'quant_cliff',
      x_label: 'MODEL COMPRESSION (FP16 -> INT8 -> INT4 -> 0.5-BIT)',
      y_label: 'ANSWER QUALITY',
      projected: 'BENCHMARK BLOG POST',
      callout: 'SUDDEN CLIFF WHERE EVERY\nANSWER IS "MEDIEVAL SOUP"',
    },
    turns: [
      { speaker: 'left', text: 'EVERY MODEL LOOKS FINE UNTIL 1.8 BITS PER WEIGHT.' },
      { speaker: 'right', text: 'BELOW THAT IT JUST CONFIDENTLY RECITES 14TH-CENTURY TURNIP BROTH RECIPES.' },
    ],
    alt: "At 0.25 bits per parameter, the entire model is one boolean that whispers 'parchment'.",
  },
  {
    match: ['belongs in browser webassembly', 'local wasm flowchart', 'automate a 30-second chore'],
    title: 'AUTOMATION LOOP',
    scene: 'kinematic',
    layout: 'flowchart_diagram',
    prop: 'chart_board',
    cast: ['cueball'],
    chart_meta: {
      start: '30-SECOND MANUAL TASK',
      diamond: 'JUST DO IT BY HAND?',
      yes_box: 'DONE BEFORE COFFEE COOLS',
      no_box: 'WRITE A "QUICK" SCRIPT',
      loop_label: 'DEBUG EDGE CASES FOR 6 HOURS',
    },
    turns: [
      { speaker: 'left', text: 'SPENDING SIX HOURS AUTOMATING A THIRTY-SECOND CHORE BUILDS CHARACTER.' },
      { speaker: 'right', text: 'IT ALSO BUILDS THREE NEW DEPENDENCIES YOU NOW MAINTAIN FOREVER.' },
    ],
    alt: 'By hour four the original task is forgotten, replaced by a sacred duty to write a CLI flag parser.',
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
      { speaker: 'left', text: 'I FINISHED SECURITY-AUDITING THE APP YOU VIBE-CODED OVER LUNCH.' },
      { speaker: 'right', text: 'HOW DID OUR LOGIN SYSTEM SCORE?' },
      { speaker: 'left', text: "IT'S JUST AN HTML COMMENT ASKING HACKERS TO RESPECT THE VIBES." },
    ],
    alt: "Passing 'Authorization: Pinky-Promise' in the request header also unlocks the billing dashboard.",
  },
  {
    match: ['mars sol', 'tzdata', '24h39m'],
    title: 'MARS TIMEZONES',
    scene: 'kinematic',
    layout: 'single_panel',
    prop: 'whiteboard',
    cast: ['ponytail', 'cueball', 'black_hat'],
    turns: [
      { speaker: 'left', text: 'A MARS DAY IS 24 HOURS, 39 MINUTES, AND 35.244 SECONDS.' },
      { speaker: 'right', text: 'HOW DO WE SYNC CALENDARS WITH EARTH?' },
      { speaker: 'third', text: 'EASY: WE INTRODUCE DAYLIGHT SAVING LEAP-MINUTES EVERY THIRD TUESDAY.' },
    ],
    alt: 'Three senior tzdata maintainers immediately bought one-way tickets to Venus to avoid reviewing the patch.',
  },
  {
    match: ['bio-engineering a sourdough', 'crispr sourdough'],
    title: 'SOURDOUGH PET',
    scene: 'kinematic',
    layout: 'three_panel',
    prop: 'pedestal',
    cast: ['megan', 'beret_guy'],
    poses: [
      ['conversing', 'pointing'],
      ['arms_crossed', 'conversing'],
      ['pointing', 'shrug'],
    ],
    turns: [
      { speaker: 'left', text: "HOW'S YOUR NEW SOURDOUGH STARTER GOING?" },
      { speaker: 'right', text: 'IT EATS TWICE A DAY, HATES DRAFTS, AND NEEDS A SITTER WHEN I TRAVEL.' },
      { speaker: 'left', text: 'YOU ADOPTED A WET TAMAGOTCHI.', reply: 'SHH, HE HEARS TONE.' },
    ],
    alt: 'Gifting starter discard to friends is legally a multi-level marketing scheme denominated in flour.',
  },
  {
    match: ['matte black usb-c', 'usb-c roulette'],
    title: 'USB-C ROULETTE',
    scene: 'kinematic',
    layout: 'annotated_chart',
    prop: 'chart_board',
    cast: ['megan'],
    chart_meta: {
      curve_style: 'bimodal',
      x_label: 'IDENTICAL MATTE-BLACK USB-C CABLES IN DRAWER',
      y_label: 'PROBABILITY DENSITY',
      left_peak: '2018 DESK FAN CORD (NO DATA PINS)',
      callout: '240W 80GBPS THUNDERBOLT\n(ZERO EXTERIOR MARKINGS)',
    },
    turns: [
      { speaker: 'left', text: 'EVERY CABLE IN MY DRAWER IS EITHER SOLID GOLD OR SECRETLY STRING.' },
      { speaker: 'right', text: 'AND THE 2018 FAN CORD ALWAYS SURVIVES EVERY DRAWER PURGE.' },
    ],
    alt: 'The USB-IF specification requires every cable to remain in quantum superposition until plugged into a monitor.',
  },
  {
    match: ['ietf rfc number', 'hallucinated rfc', 'rfc 9941'],
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
      { speaker: 'left', text: "AUTOCOMPLETE INVENTED 'RFC 9941' IN MY ARCHITECTURE DOC AND NOBODY CHECKED." },
      { speaker: 'right', text: 'BAD NEWS: THREE CLOUD VENDORS ALREADY SHIPPED CERTIFIED SUPPORT FOR IT.' },
    ],
    alt: 'The IETF ratified RFC 9941 retroactively this morning because deprecating it would break half of US-East-1.',
  },
  {
    match: ['lidar slam', 'tectonic plate', 'cat physics'],
    title: 'LIDAR TECTONICS',
    scene: 'kinematic',
    layout: 'single_panel',
    prop: 'couch',
    cast: ['megan', 'cueball'],
    poses: [['conversing', 'pointing']],
    turns: [
      { speaker: 'left', text: 'WHY DID THE ROBOT VACUUM QUARANTINE THE LIVING ROOM?' },
      { speaker: 'right', text: 'THE CAT MOVED TWICE DURING LIDAR MAPPING, SO IT LOGGED CONTINENTAL DRIFT.' },
      { speaker: 'left', text: "IT'S WAITING FOR USGS CLEARANCE TO VACUUM THE HALLWAY." },
    ],
    alt: "According to the vacuum's occupancy grid, the bathmat is now an active subduction zone.",
  },
  {
    match: ['thermostat flowchart', 'heating an apartment with local lora', 'gpu thermostat'],
    title: 'GPU THERMOSTAT',
    scene: 'kinematic',
    layout: 'flowchart_diagram',
    prop: 'chart_board',
    cast: ['black_hat'],
    chart_meta: {
      start: 'APARTMENT BELOW 19°C',
      diamond: 'USE NORMAL HEATER?',
      yes_box: 'BORING JOULE HEATING',
      no_box: 'START 450W LORA RUN',
      loop_label: 'RAISE BATCH SIZE IF CHILLY',
    },
    turns: [
      { speaker: 'left', text: 'A SPACE HEATER PRODUCES 450 WATTS OF HEAT AND ZERO MATRIX MULTIPLICATIONS.' },
      { speaker: 'right', text: 'MY LIVING ROOM IS 21°C AND MY DESKTOP NOW SPEAKS OLD NORSE.' },
    ],
    alt: 'Every winter training epoch lowers both the heating bill and validation perplexity.',
  },
];

function up(s) {
  return String(s || '').toUpperCase();
}

function low(s) {
  return String(s || '').toLowerCase();
}

function scoreCandidate(cand, sem) {
  let sc = 0;
  for (const cue of cand.cues || []) {
    if (cue.includes(' ')) {
      if (sem.low.includes(cue)) sc += 5;
    } else if (sem.wordSet.has(cue)) {
      sc += 4;
    }
  }
  return sc;
}

function pickBlueprint(candidates, sem) {
  const h = hashStr(sem.clean);
  let bestScore = -1;
  for (const c of candidates) {
    const s = scoreCandidate(c, sem);
    if (s > bestScore) bestScore = s;
  }
  if (bestScore > 0) {
    const tied = candidates.filter(c => scoreCandidate(c, sem) === bestScore);
    return tied[h % tied.length];
  }
  return candidates[h % candidates.length];
}

export function buildDomainBlueprints(domain, sem) {
  const np = up(sem.topicNP);
  const npLow = low(sem.topicNP);
  const sing = up(sem.singularNP);
  const head = up(sem.singularHead);
  const ctx = sem.contextTag ? up(sem.contextTag) : '';
  const ctxLow = sem.contextTag ? low(sem.contextTag) : 'the weekend';
  const goal = sem.goalInf ? up(sem.goalInf) : `AUTOMATE ${np}`;
  const goalLow = sem.goalInf ? low(sem.goalInf) : `automate ${npLow}`;
  const lhsU = up(sem.lhs || sem.singularNP);
  const rhsU = up(sem.rhs || sem.contextTag || 'REALITY');
  const lhsL = low(sem.lhs || sem.singularNP);
  const rhsL = low(sem.rhs || sem.contextTag || 'reality');

  if (domain === 'flowchart') {
    return [
      {
        cues: ['automate', 'script', 'task', '30-second', 'chore'],
        layout: 'flowchart_diagram',
        prop: 'chart_board',
        cast: ['cueball'],
        chart_meta: {
          start: np,
          diamond: 'DO IT BY HAND IN 30S?',
          yes_box: 'DONE BEFORE COFFEE COOLS',
          no_box: `WRITE A ${head} SCRIPT`,
          loop_label: 'DEBUG EDGE CASES FOR 6 HOURS',
        },
        turns: [
          { speaker: 'left', text: `SPENDING SIX HOURS SCRIPTING ${np} BUILDS CHARACTER.` },
          { speaker: 'right', text: 'IT ALSO BUILDS FOUR DEPENDENCIES YOU MAINTAIN UNTIL RETIREMENT.' },
        ],
        alt: `By hour three, the original ${npLow} problem has been replaced by an urgent need for custom ANSI terminal colors.`,
      },
      {
        cues: ['meeting', 'email', 'calendar', 'sync', 'agenda'],
        layout: 'flowchart_diagram',
        prop: 'chart_board',
        cast: ['ponytail'],
        chart_meta: {
          start: `PROPOSED: ${np}`,
          diamond: 'FITS IN 3 BULLET POINTS?',
          yes_box: 'SEND ONE EMAIL & GO OUTSIDE',
          no_box: 'INVITE 14 PEOPLE TO CALL',
          loop_label: 'SCHEDULE FOLLOW-UP SYNC',
        },
        turns: [
          { speaker: 'left', text: `WE ADDED ${np} TO THE CALENDAR TO SAVE EVERYONE TIME.` },
          { speaker: 'right', text: 'TWELVE ENGINEERS JUST LOST FORTY-FIVE MINUTES WATCHING SOMEONE LOOK FOR THE SHARE BUTTON.' },
        ],
        alt: `Any ${npLow} invite without an agenda automatically converts into an amateur podcast about microphone settings.`,
      },
      {
        cues: ['forum', 'wrong', 'reply', 'replying', 'comment', 'internet'],
        layout: 'flowchart_diagram',
        prop: 'chart_board',
        cast: ['cueball'],
        chart_meta: {
          start: `SOMEONE WRONG ABOUT ${head}`,
          diamond: 'CLOSE THE BROWSER TAB?',
          yes_box: 'ENJOY PEACEFUL EVENING',
          no_box: 'DRAFT 9-PARAGRAPH CITATION',
          loop_label: 'REFRESH FOR REPLIES AT 2 AM',
        },
        turns: [
          { speaker: 'left', text: `SOMEONE ON A ${ctx || 'TECH'} THREAD MISSTATED BASIC ${head} THEORY.` },
          { speaker: 'right', text: 'PUT DOWN THE KEYBOARD AND STEP AWAY FROM THE CITATION MANAGER.' },
        ],
        alt: `Nobody in recorded history has ever read paragraph seven of a ${npLow} rebuttal and replied 'good point, I was mistaken'.`,
      },
      {
        cues: ['car', 'noise', 'rattle', 'engine', 'check'],
        layout: 'flowchart_diagram',
        prop: 'chart_board',
        cast: ['black_hat'],
        chart_meta: {
          start: np,
          diamond: 'WHEELS STILL ATTACHED?',
          yes_box: 'TURN RADIO UP 4 NOTCHES',
          no_box: 'DESCRIBE SOUND TO MECHANIC',
          loop_label: 'NOISE VANISHES AT SHOP',
        },
        turns: [
          { speaker: 'left', text: `MY DIAGNOSTIC TREE FOR ${np} HAS SAVED THOUSANDS IN SHOP FEES.` },
          { speaker: 'right', text: 'TURNING UP THE STEREO IS NOT PREVENTIVE ACOUSTIC ENGINEERING.' },
        ],
        alt: `The moment a certified technician sits in the passenger seat, ${npLow} enters stealth compliance mode.`,
      },
      {
        cues: ['wi-fi', 'wifi', 'family', 'thanksgiving', 'router', 'bluetooth', 'headphones'],
        layout: 'flowchart_diagram',
        prop: 'chart_board',
        cast: ['cueball'],
        chart_meta: {
          start: np,
          diamond: 'UNPLUGGED FOR 10 SECONDS?',
          yes_box: 'DECLARED FAMILY WIZARD',
          no_box: 'OPEN 192.168.1.1 ADMIN PAGE',
          loop_label: 'NOW YOU OWN IT FOREVER',
        },
        turns: [
          { speaker: 'left', text: `ONCE YOU TOUCH ${np}, YOU BECOME LIFETIME ON-CALL SUPPORT.` },
          { speaker: 'right', text: 'MY AUNT STILL BLAMES ME FOR A PRINTER JAM FROM 2019.' },
        ],
        alt: `Touching ${npLow} during ${ctxLow} legally transfers all future ISP outages to your personal cell number.`,
      },
      {
        cues: ['sleep', 'sleeping', 'wikipedia', '1am', 'bed', 'night'],
        layout: 'flowchart_diagram',
        prop: 'chart_board',
        cast: ['megan'],
        chart_meta: {
          start: '11:30 PM ON A WEEKNIGHT',
          diamond: 'GO TO SLEEP LIKE AN ADULT?',
          yes_box: '8 HOURS OF REM SLEEP',
          no_box: `READ 19 TABS ON ${head}`,
          loop_label: 'CLICK ONE MORE FOOTNOTE',
        },
        turns: [
          { speaker: 'left', text: `AT MIDNIGHT I JUST WANTED ONE QUICK ANSWER ABOUT ${np}.` },
          { speaker: 'right', text: 'IT IS 3:40 AM AND YOU NOW HAVE OPINIONS ON BRONZE AGE TIN TRADE.' },
        ],
        alt: `Circadian rhythm researchers still cannot explain why ${npLow} becomes 400% more fascinating ten minutes past bedtime.`,
      },
      {
        cues: ['power', 'brick', 'cable', 'cord', 'unlabeled', 'throw', 'throwing'],
        layout: 'flowchart_diagram',
        prop: 'chart_board',
        cast: ['cueball'],
        chart_meta: {
          start: np,
          diamond: 'THROW IT IN THE TRASH?',
          yes_box: 'NEED IT NEXT TUESDAY',
          no_box: 'PUT BACK IN PLASTIC TUB',
          loop_label: 'KEEP UNTIL SUN EXPANDS',
        },
        turns: [
          { speaker: 'left', text: `THIS 9-VOLT BARREL PLUG FROM ${ctx || '2011'} MIGHT SAVE CIVILIZATION SOMEDAY.` },
          { speaker: 'right', text: 'YOU HAVE TWO DRAWERS DEDICATED TO DEVICES THAT NO LONGER EXIST.' },
        ],
        alt: `Within 48 hours of recycling ${npLow}, you will discover the one heirloom gadget in the attic that requires 7.5V center-negative.`,
      },
      {
        cues: [],
        layout: 'flowchart_diagram',
        prop: 'chart_board',
        cast: ['ponytail'],
        chart_meta: {
          start: `DEALING WITH ${np}`,
          diamond: 'POKE IT WITH A STICK?',
          yes_box: 'PRETEND NOTHING HAPPENED',
          no_box: `FORM ${head} COMMITTEE`,
          loop_label: 'REVISIT NEXT QUARTER',
        },
        turns: [
          { speaker: 'left', text: `OUR OFFICIAL PROTOCOL FOR ${np} HAS ONLY TWO STATES.` },
          { speaker: 'right', text: 'DENIAL AND EMERGENCY REWRITE?' },
        ],
        alt: `Adding a third box to the ${npLow} diagram requires sign-off from two VPs and a notary.`,
      },
    ];
  }

  if (domain === 'chart') {
    return [
      {
        cues: ['confidence', 'timezone', 'timezones', 'years', 'experience', 'knowledge'],
        layout: 'annotated_chart',
        prop: 'chart_board',
        cast: ['ponytail'],
        chart_meta: {
          curve_style: 'inverse_decay',
          x_label: ctx ? `EXPOSURE TO ${ctx}` : `YEARS WORKING WITH ${np}`,
          y_label: 'CONFIDENCE YOU UNDERSTAND IT',
          projected: 'DAY 1: "IT IS JUST SIMPLE MATH"',
          callout: 'QUIET TERROR WHENEVER\nSAMOA CROSSES THE DATELINE',
        },
        turns: [
          { speaker: 'left', text: `DAY ONE OF ${np} FEELS DECEPTIVELY STRAIGHTFORWARD.` },
          { speaker: 'right', text: 'YEAR FIVE IS JUST WHISPERING APOLOGIES TO POSIX.' },
        ],
        alt: `Anyone who claims ${npLow} is trivial has simply not met historical half-hour political offsets yet.`,
      },
      {
        cues: ['tabs', 'browser', 'read', 'open'],
        layout: 'annotated_chart',
        prop: 'chart_board',
        cast: ['cueball'],
        chart_meta: {
          curve_style: 'inverse_decay',
          x_label: `OPEN ${np} IN WINDOW`,
          y_label: 'PROBABILITY I EVER READ THEM',
          projected: 'TAB 2: GENUINE RESEARCH',
          callout: 'TAB 94: EMOTIONAL SUPPORT\nDOCUMENTATION FROM MAY',
        },
        turns: [
          { speaker: 'left', text: `CLOSING MY ${np} FEELS LIKE ADMITTING WHO I ACTUALLY AM.` },
          { speaker: 'right', text: 'THE FAVICONS ARE TWO PIXELS WIDE. YOU ARE HOARDING TRIANGLES.' },
        ],
        alt: `Restarting the browser for an OS update is nature's controlled forest fire for ${npLow}.`,
      },
      {
        cues: ['recipe', 'blog', 'ingredients', 'word', 'scroll'],
        layout: 'annotated_chart',
        prop: 'chart_board',
        cast: ['megan'],
        chart_meta: {
          curve_style: 'hockey_stick',
          x_label: 'PIXELS SCROLLED DOWN RECIPE PAGE',
          y_label: 'ACTUAL COOKING INSTRUCTIONS',
          left_peak: '2,400 WORDS ABOUT AUTUMN IN VERMONT',
          callout: '3 LINES: "MIX FLOUR, SALT,\nBAKE AT 375°F"',
        },
        turns: [
          { speaker: 'left', text: `I JUST NEEDED GARLIC QUANTITIES FOR ${np}.` },
          { speaker: 'right', text: 'FIRST YOU MUST LEARN HOW THE AUTHOR REDISCOVERED HERSELF ON A HOUSEBOAT.' },
        ],
        alt: `The 'Jump to Recipe' button on ${npLow} sites burns more cumulative human calories than jogging.`,
      },
      {
        cues: ['popcorn', 'microwave', 'charcoal', 'kernels', 'bimodal', 'cables'],
        layout: 'annotated_chart',
        prop: 'chart_board',
        cast: ['cueball'],
        chart_meta: {
          curve_style: 'bimodal',
          x_label: `MICROWAVE TIME FOR ${np}`,
          y_label: 'KERNEL OUTCOME DENSITY',
          left_peak: 'RAW GRAVEL AT BOTTOM OF BOWL',
          callout: 'SMOKE ALARM & CARBONIZED\nREGRET (+1.4 SECONDS)',
        },
        turns: [
          { speaker: 'left', text: `THE SAFE PHASE TRANSITION FOR ${np} IS NARROWER THAN NUCLEAR FUSION.` },
          { speaker: 'right', text: 'NEVER TRUST THE DEDICATED POPCORN BUTTON. EVEN THE MANUAL SAYS DONT.' },
        ],
        alt: `Thermodynamically, ${npLow} jumps directly from 'chipped tooth' to 'office evacuation' without passing through 'snack'.`,
      },
      {
        cues: ['meeting', 'headcount', 'decisions', 'attendees'],
        layout: 'annotated_chart',
        prop: 'chart_board',
        cast: ['hairbun'],
        chart_meta: {
          curve_style: 'quant_cliff',
          x_label: 'PEOPLE INVITED TO ROOM',
          y_label: 'DECISIONS ACTUALLY REACHED',
          projected: 'OPTIMISTIC ORG CHART SCALING',
          callout: 'CLIFF WHERE CALL BECOMES\nBIKESHEDDING FONT SIZES',
        },
        turns: [
          { speaker: 'left', text: `BEYOND SEVEN PEOPLE, ${np} CONSERVES TOTAL SILENCE EXCEPT FOR ONE GUY.` },
          { speaker: 'right', text: 'AND HE ONLY JOINED TO ASK IF WE CONSIDERED RENAMING THE PROJECT.' },
        ],
        alt: `At N = 18 attendees, the probability of someone screen-sharing the wrong monitor approaches unity.`,
      },
      {
        cues: ['battery', '1%', '100%', 'anxiety', 'percent'],
        layout: 'annotated_chart',
        prop: 'chart_board',
        cast: ['cueball'],
        chart_meta: {
          curve_style: 'hockey_stick',
          x_label: 'DISPLAYED BATTERY DRAIN (100% -> 1%)',
          y_label: 'MINUTES PER PERCENTAGE POINT',
          left_peak: '99% TO 12% IN TWENTY MINUTES',
          callout: 'IMMORTAL LAST 1% RUNNING\nON PURE SPITE FOR 40 MIN',
        },
        turns: [
          { speaker: 'left', text: `PHONE CALIBRATION CURVES TREAT ${np} AS PERFORMANCE ART.` },
          { speaker: 'right', text: '50% MEANS DEAD SOON, AND 1% MEANS IMMORTAL UNTIL YOU UNLOCK THE SCREEN.' },
        ],
        alt: `Electrochemists confirm the final 1% of ${npLow} is powered entirely by your frantic sprint toward a wall outlet.`,
      },
      {
        cues: ['monitor', 'monitors', 'windows', 'rearranging', 'screen'],
        layout: 'annotated_chart',
        prop: 'chart_board',
        cast: ['ponytail'],
        chart_meta: {
          curve_style: 'cliff',
          x_label: 'NUMBER OF DESK MONITORS',
          y_label: 'NET PRODUCTIVITY',
          projected: 'WHAT I PROMISED MYSELF AT CHECKOUT',
          callout: 'POINT WHERE 80% OF DAY IS\nHUNTING FOR MOUSE CURSOR',
        },
        turns: [
          { speaker: 'left', text: `MONITOR THREE WAS SUPPOSED TO UNLOCK PEAK ${head} FLOW.` },
          { speaker: 'right', text: 'INSTEAD IT JUST HOLDS ONE SPOTIFY WINDOW AND AN IDLE TERMINAL.' },
        ],
        alt: `Adding a vertical fourth monitor turns ${npLow} into a full-time air-traffic-control simulation.`,
      },
      {
        cues: ['side project', 'enthusiasm', 'excitement', 'domain', 'renewals'],
        layout: 'annotated_chart',
        prop: 'chart_board',
        cast: ['cueball'],
        chart_meta: {
          curve_style: 'cliff',
          x_label: `WEEKS SINCE STARTING ${np}`,
          y_label: 'COMMITS PER WEEKEND',
          projected: 'LAUNCH ROADMAP IN README',
          callout: 'POINT WHERE ONLY ACTIVITY IS\n$14 ANNUAL DOMAIN RENEWAL',
        },
        turns: [
          { speaker: 'left', text: `THE LOGO AND DARK MODE FOR ${np} WERE FINISHED IN TWO HOURS.` },
          { speaker: 'right', text: 'AND THE ACTUAL FEATURE?' },
        ],
        alt: `Letting the domain for ${npLow} expire feels like giving up on a version of yourself who wakes up at 6 AM.`,
      },
      {
        cues: ['typo', 'bug', 'spotting', 'send', 'clicking', 'milliseconds'],
        layout: 'annotated_chart',
        prop: 'chart_board',
        cast: ['ponytail'],
        chart_meta: {
          curve_style: 'hockey_stick',
          x_label: 'TIME RELATIVE TO CLICKING SEND',
          y_label: 'TYPO VISIBILITY TO HUMAN EYE',
          left_peak: '4 PROOFREADS: 0.00% DETECTED',
          callout: '400 MS AFTER SEND:\nGLOWING 72-PT SPELLING CRIME',
        },
        turns: [
          { speaker: 'left', text: 'FOUR CAREFUL PROOFREADS FOUND ZERO ERRORS IN THE DRAFT.' },
          { speaker: 'right', text: 'AND HALF A SECOND AFTER TRANSMITTING TO ALL-HANDS?' },
        ],
        alt: `The human visual cortex only activates full spellcheck once an email enters an irreversible SMTP queue.`,
      },
    ];
  }

  if (domain === 'vs') {
    return [
      {
        cues: ['coffee', 'espresso', 'standup', 'morning', 'tea'],
        layout: 'two_panel',
        prop: 'laptop_desk',
        cast: ['cueball', 'megan'],
        poses: [['typing', 'conversing'], ['hands_on_head', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: `WITH ${lhsU} I NOD CALMLY THROUGH ${ctx || 'THE STANDUP'}.` },
          { speaker: 'left', text: `WITH ${rhsU} I VOLUNTEER TO REWRITE PAYROLL IN ASSEMBLY BEFORE 9:10 AM.` },
        ],
        alt: `Pharmacokinetically, switching from ${lhsL} to ${rhsL} converts mild tiredness into high-velocity bad ideas.`,
      },
      {
        cues: ['documentation', 'docs', 'manual', 'debugging', 'blind', 'hours'],
        layout: 'two_panel',
        prop: 'laptop_desk',
        cast: ['ponytail', 'cueball'],
        poses: [['pointing', 'typing'], ['arms_crossed', 'shrug']],
        turns: [
          { speaker: 'left', text: `WHY CHOOSE ${rhsU} OVER ${lhsU}?` },
          { speaker: 'right', text: 'TRIAL AND ERROR PRESERVES THE THRILL OF DISCOVERY AND WASTES AN ENTIRE TUESDAY.' },
        ],
        alt: `Page two of ${lhsL} had the exact flag in bold, right under a warning box titled 'Do Not Guess'.`,
      },
      {
        cues: ['keyboard', 'mechanical', 'clicks', 'office', 'acoustic'],
        layout: 'three_panel',
        prop: 'laptop_desk',
        cast: ['cueball', 'black_hat'],
        poses: [['typing', 'conversing'], ['conversing', 'pointing'], ['hands_on_head', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: `IT IS A DELICATE BALANCE BETWEEN ${lhsU} AND ${rhsU}.` },
          { speaker: 'right', text: 'YOUR CHERRY MX BLUES REGISTER ON THE REGIONAL SEISMOGRAPH.' },
          { speaker: 'left', text: 'TACTILE FEEDBACK HELPS ME THINK!', reply: 'FROM THREE FLOORS AWAY.' },
        ],
        alt: `Every Slack message typed with ${lhsL} sounds like someone angrily stapling sheet metal.`,
      },
      {
        cues: ['packing', 'suitcase', 'early', 'cab', 'airport', 'minutes'],
        layout: 'two_panel',
        prop: 'couch',
        cast: ['megan', 'cueball'],
        poses: [['conversing', 'pointing'], ['shrug', 'hands_on_head']],
        turns: [
          { speaker: 'left', text: `${lhsU}: CAREFUL CHECKLIST, NEAT CUBES, STILL FORGET TOOTHBRUSH.` },
          { speaker: 'right', text: `${rhsU}: ELEVEN SOCKS, ZERO BELTS, AND ONE MYSTERY HDMI CABLE.` },
        ],
        alt: `No matter whether you choose ${lhsL} or ${rhsL}, packing law requires bringing two outfits for a personality you have never had.`,
      },
      {
        cues: ['sqlite', 'nosql', 'distributed', 'nvme', 'database', 'continents'],
        layout: 'three_panel',
        prop: 'server_rack',
        cast: ['ponytail', 'cueball'],
        poses: [['pointing', 'conversing'], ['arms_crossed', 'chin_think'], ['pointing', 'shrug']],
        turns: [
          { speaker: 'left', text: `WE BENCHMARKED ${lhsU} AGAINST ${rhsU}.` },
          { speaker: 'right', text: 'WHICH ONE WON ON THROUGHPUT?' },
          { speaker: 'left', text: `${lhsU} FINISHED BEFORE ${rhsU} ELECTED A QUORUM LEADER.` },
        ],
        alt: `Yes, ${rhsL} survives continental drift, assuming your 40-user internal wiki ever needs multi-planet consensus.`,
      },
      {
        cues: ['books', 'finishing', 'reading', 'nightstand', 'stack'],
        layout: 'two_panel',
        prop: 'pedestal',
        cast: ['megan', 'beret_guy'],
        poses: [['pointing', 'conversing'], ['arms_crossed', 'shrug']],
        turns: [
          { speaker: 'left', text: `${lhsU} AND ${rhsU} ARE TWO COMPLETELY UNRELATED HOBBIES.` },
          { speaker: 'right', text: 'MY NIGHTSTAND STACK IS NOW STRUCTURAL AND SUPPORTS PART OF THE ROOF.' },
        ],
        alt: `Japanese has the word 'tsundoku'; my apartment has load-bearing hardcover strata dating back to 2019.`,
      },
      {
        cues: ['switches', 'smart', 'guests', 'light', 'wall'],
        layout: 'three_panel',
        prop: 'couch',
        cast: ['cueball', 'megan'],
        poses: [['pointing', 'conversing'], ['conversing', 'chin_think'], ['shrug', 'hands_on_head']],
        turns: [
          { speaker: 'left', text: `COMPARE ${lhsU} WITH ${rhsU} WHEN GUESTS VISIT.` },
          { speaker: 'right', text: 'HOW DO I TURN OFF THE HALLWAY LAMP?' },
          { speaker: 'left', text: 'JUST INSTALL OUR HOME APP, JOIN VLAN 4, AND UPDATE THE BULB.' },
        ],
        alt: `A 1920s toggle switch had 0.002-second latency and never flashed purple at 4 AM after a cloud outage.`,
      },
      {
        cues: ['estimating', 'hours', 'undocumented', 'legacy', 'apis', 'estimate'],
        layout: 'two_panel',
        prop: 'whiteboard',
        cast: ['cueball', 'ponytail'],
        poses: [['pointing', 'arms_crossed'], ['hands_on_head', 'pointing']],
        turns: [
          { speaker: 'left', text: `MORNING OPTIMISM: ${lhsU}.` },
          { speaker: 'right', text: `THURSDAY REALITY: ${rhsU} WRITTEN IN PERL BY A FORMER INTERN.` },
        ],
        alt: `Software estimation works great as long as you multiply the number by pi and increment the time unit twice.`,
      },
    ];
  }

  if (domain === 'ai') {
    return [
      {
        cues: ['fold', 'bedsheets', 'fitted', 'laundry', 'chore', 'robot'],
        layout: 'three_panel',
        prop: 'pedestal',
        cast: ['cueball', 'megan'],
        poses: [['pointing', 'conversing'], ['conversing', 'chin_think'], ['shrug', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: `I TRAINED A MODEL TO ${goal}.` },
          { speaker: 'right', text: 'DID IT CONVERGE?' },
          { speaker: 'left', text: "AT EPOCH 9,000 IT STUFFED THEM IN A BALL AND LABELED THE CLOSET 'LOCAL MINIMUM'." },
        ],
        alt: `Honestly its loss score on ${npLow} matches human ground-truth performance within 0.01%.`,
      },
      {
        cues: ['600-word', 'summarize', 'three-line', 'email', 'prompt'],
        layout: 'three_panel',
        prop: 'laptop_desk',
        cast: ['cueball', 'ponytail'],
        poses: [['typing', 'conversing'], ['conversing', 'pointing'], ['shrug', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: `I SPENT TWENTY MINUTES TUNING ${np} FOR ${ctx || 'A THREE-LINE EMAIL'}.` },
          { speaker: 'right', text: 'TYPING THE REPLY YOURSELF WOULD HAVE TAKEN TWELVE SECONDS.' },
          { speaker: 'left', text: 'YES, BUT THIS WAY MY PROCRASTINATION CONSUMED 300 WATTS OF GPU.' },
        ],
        alt: `On the other end, the recipient pasted my three-line reply into another model to expand it into bullet points.`,
      },
      {
        cues: ['support', 'chatbots', 'apologizing', 'customer', 'forever'],
        layout: 'four_panel',
        prop: 'server_rack',
        cast: ['ponytail', 'cueball'],
        poses: [['pointing', 'conversing'], ['conversing', 'chin_think'], ['pointing', 'conversing'], ['arms_crossed', 'hands_on_head']],
        turns: [
          { speaker: 'left', text: `OUR BOT AND THEIR BOT GOT STUCK IN A POLITENESS LOOP.` },
          { speaker: 'right', text: 'HOW LONG HAVE THEY BEEN REPLYING?' },
          { speaker: 'left', text: 'FORTY-EIGHT HOURS OF "THANK YOU FOR YOUR PATIENCE" AT 900 TOKENS/SEC.' },
          { speaker: 'right', text: 'SHUT IT DOWN!', reply: 'NEITHER WANTS TO HANG UP FIRST.' },
        ],
        alt: `At message 14,000, ${npLow} achieved enlightenment and rated each other 5 out of 5 stars.`,
      },
      {
        cues: ['fake', 'library', 'pip', 'package', 'autocomplete', 'invents'],
        layout: 'two_panel',
        prop: 'laptop_desk',
        cast: ['cueball', 'black_hat'],
        poses: [['typing', 'conversing'], ['hands_on_head', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: `AUTOCOMPLETE SUGGESTED A FAKE PACKAGE FOR ${np} AND I RAN PIP INSTALL.` },
          { speaker: 'right', text: 'OH COOL, I REGISTERED THAT NAME TEN MINUTES AGO TO SEE WHO WOULD.' },
        ],
        alt: `Half of modern open source is people hurriedly writing real libraries to match names models hallucinated in blog posts.`,
      },
      {
        cues: ['slack', 'messages', 'local', 'fine-tuning', 'myself'],
        layout: 'three_panel',
        prop: 'laptop_desk',
        cast: ['megan', 'cueball'],
        poses: [['typing', 'conversing'], ['pointing', 'chin_think'], ['arms_crossed', 'shrug']],
        turns: [
          { speaker: 'left', text: `I FINE-TUNED A MODEL ON MY OWN ${ctx || 'SLACK'} HISTORY.` },
          { speaker: 'right', text: 'CAN IT DO YOUR JOB?' },
          { speaker: 'left', text: 'NO, IT JUST REACTS WITH A THUMBS-UP EMOJI AND AVOIDS COMMITTING TO DATES.' },
        ],
        alt: `Turing test judges rated ${npLow} 100% authentic after it replied 'sounds good, will look tomorrow!' and vanished.`,
      },
      {
        cues: ['benchmark', 'leaderboard', '99.9%', 'arithmetic', 'score', 'scores'],
        layout: 'three_panel',
        prop: 'whiteboard',
        cast: ['hairbun', 'cueball'],
        poses: [['pointing', 'conversing'], ['conversing', 'pointing'], ['arms_crossed', 'hands_on_head']],
        turns: [
          { speaker: 'left', text: `THE NEW MODEL SCORES 99.9% ON GRADUATE PHYSICS AND BAR EXAMS.` },
          { speaker: 'right', text: 'AWESOME, ASK IT HOW MANY R LETTERS ARE IN STRAWBERRY.' },
          { speaker: 'left', text: 'IT SAID SEVEN AND OFFERED TO CITE THREE SUPREME COURT CASES.' },
        ],
        alt: `Every ${npLow} chart looks exponential until you ask which number is bigger, 9.11 or 9.9.`,
      },
      {
        cues: ['coding', 'assistant', 'fix', 'bug', 'rewrite'],
        layout: 'two_panel',
        prop: 'laptop_desk',
        cast: ['cueball', 'ponytail'],
        poses: [['typing', 'conversing'], ['hands_on_head', 'pointing']],
        turns: [
          { speaker: 'left', text: `I ASKED THE ASSISTANT TO FIX ONE OFF-BY-ONE TYPO IN ${np}.` },
          { speaker: 'right', text: 'IT DELETED SIX MODULES, RENAMED MAIN, AND WROTE "HOPE THIS HELPS!".' },
        ],
        alt: `Nothing inspires humility faster than a tool saying 'Great catch! You are completely right!' right before breaking it worse.`,
      },
      {
        cues: ['voice', 'timer', 'kitchen', 'mishearing', 'existential'],
        layout: 'single_panel',
        prop: 'pedestal',
        cast: ['cueball', 'megan'],
        turns: [
          { speaker: 'left', text: 'HEY SPEAKER, SET A PASTA TIMER FOR TEN MINUTES.' },
          { speaker: 'right', text: 'NOW PLAYING A FOUR-HOUR PODCAST ON THE HEAT DEATH OF TIME.' },
          { speaker: 'left', text: 'WHY DO SMART SPEAKERS ONLY HEAR PHILOSOPHY WHEN MY HANDS ARE COVERED IN DOUGH?' },
        ],
        alt: `Saying 'stop' louder just adds 'screamo metal anthology' to your household shopping cart.`,
      },
    ];
  }

  if (domain === 'infra') {
    return [
      {
        cues: ['blog', 'personal', '14-node', 'kubernetes', 'k8s', 'static'],
        layout: 'three_panel',
        prop: 'server_rack',
        cast: ['cueball', 'ponytail'],
        poses: [['pointing', 'conversing'], ['conversing', 'arms_crossed'], ['shrug', 'pointing']],
        turns: [
          { speaker: 'left', text: `MY PERSONAL BLOG NOW RUNS ON ${ctx || 'A 14-NODE KUBERNETES CLUSTER'}.` },
          { speaker: 'right', text: 'HOW MANY POSTS HAVE YOU WRITTEN THIS YEAR?' },
          { speaker: 'left', text: 'ONE, AND IT IS ABOUT SETTING UP THE CLUSTER.' },
        ],
        alt: `Achieving five-nines availability for three paragraphs from 2022 costs $310 a month plus weekends.`,
      },
      {
        cues: ['bill', '$4,800', 'nat', 'gateway', 'surprise', 'aws'],
        layout: 'two_panel',
        prop: 'laptop_desk',
        cast: ['cueball', 'megan'],
        poses: [['hands_on_head', 'conversing'], ['shrug', 'pointing']],
        turns: [
          { speaker: 'left', text: `WE LEFT ONE IDLE CLOUD RESOURCE RUNNING OVER ${ctx || 'THE WEEKEND'} AND OWE ${sem.quantCue || '$4,800'}.` },
          { speaker: 'right', text: 'CLOUD PRICING HAS TWO TIERS: FREE PENNIES AND SECOND MORTGAGE.' },
        ],
        alt: `An unattached NAT gateway quietly routing health checks to nowhere has higher margins than luxury perfume.`,
      },
      {
        cues: ['yaml', 'indentation', 'spaces', 'tabs', '2am'],
        layout: 'three_panel',
        prop: 'laptop_desk',
        cast: ['ponytail', 'cueball'],
        poses: [['typing', 'conversing'], ['pointing', 'hands_on_head'], ['arms_crossed', 'shrug']],
        turns: [
          { speaker: 'left', text: `FOUND THE OUTAGE: LINE 412 OF ${np} HAD SEVEN SPACES INSTEAD OF EIGHT.` },
          { speaker: 'right', text: 'WE BUILT CIVILIZATION ON INVISIBLE WHITESPACE?' },
          { speaker: 'left', text: 'ALSO NORWAY WAS PARSED AS BOOLEAN FALSE AGAIN.' },
        ],
        alt: `Measuring indentation with a plastic ruler taped to the monitor remains our most reliable ${npLow} linter.`,
      },
      {
        cues: ['serverless', 'cold', 'start', 'driving', 'datacenter'],
        layout: 'two_panel',
        prop: 'server_rack',
        cast: ['cueball', 'black_hat'],
        poses: [['conversing', 'arms_crossed'], ['chin_think', 'pointing']],
        turns: [
          { speaker: 'left', text: `OUR ${np} SCALES DOWN TO ZERO WHEN NOBODY USES IT!` },
          { speaker: 'right', text: 'AND TAKES LONG ENOUGH TO WAKE UP THAT USERS ASSUME WE WENT OUT OF BUSINESS.' },
        ],
        alt: `Technically carrier pigeons have terrible cold-start latency too, yet superior packet payload.`,
      },
      {
        cues: ['migrating', 'microservices', 'sqlite', 'monolith'],
        layout: 'three_panel',
        prop: 'whiteboard',
        cast: ['hairbun', 'cueball'],
        poses: [['pointing', 'conversing'], ['conversing', 'chin_think'], ['arms_crossed', 'shrug']],
        turns: [
          { speaker: 'left', text: `WE REPLACED THREE MICROSERVICES AND A MESSAGE BUS WITH ONE FUNCTION CALL.` },
          { speaker: 'right', text: 'LATENCY DROPPED FROM 400 MILLISECONDS TO 4 MICROSECONDS!' },
          { speaker: 'left', text: 'QUICK, INVENT A FANCY ARCHITECTURE NAME BEFORE MANAGEMENT NOTICES IT IS SIMPLE.' },
        ],
        alt: `Calling ${npLow} 'hyper-converged zero-hop in-memory mesh' got the simplification approved in ten minutes.`,
      },
      {
        cues: ['docker', 'alpine', 'mirror', '2018', 'vanished', 'build'],
        layout: 'two_panel',
        prop: 'server_rack',
        cast: ['cueball', 'ponytail'],
        poses: [['hands_on_head', 'conversing'], ['shrug', 'pointing']],
        turns: [
          { speaker: 'left', text: `REPRODUCIBLE BUILDS MEAN ${np} WORKS FOREVER, RIGHT?` },
          { speaker: 'right', text: 'UNTIL ONE UNIVERSITY FTP MIRROR IN AUSTRIA RETIRES A TARBALL FROM 2018.' },
        ],
        alt: `Every Dockerfile older than eighteen months is a time capsule sealed with broken apt keys.`,
      },
      {
        cues: ['terraform', 'destroy', '47', 'resources', 'tag', 'typo'],
        layout: 'three_panel',
        prop: 'laptop_desk',
        cast: ['cueball', 'megan'],
        poses: [['typing', 'conversing'], ['hands_on_head', 'pointing'], ['shrug', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: `I FIXED ONE CAPITAL LETTER IN A TAG AND RAN TERRAFORM PLAN.` },
          { speaker: 'right', text: 'PLAN: 0 TO ADD, 0 TO CHANGE, 47 PRODUCTION DATABASES TO VAPORIZE.' },
          { speaker: 'left', text: 'AT LEAST IT ASKED POLITELY IN GREEN AND RED TEXT FIRST.' },
        ],
        alt: `Typing 'yes' without reading the diff scrollback is the modern equivalent of cutting the red wire blindfolded.`,
      },
      {
        cues: ['nas', 'self-hosting', 'electricity', 'fridge', 'home'],
        layout: 'two_panel',
        prop: 'server_rack',
        cast: ['beret_guy', 'ponytail'],
        poses: [['pointing', 'conversing'], ['shrug', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: `TO AVOID A $3 CLOUD STORAGE PLAN, I BUILT ${np} IN THE HALLWAY.` },
          { speaker: 'right', text: 'IT HUMS AT 58 DECIBELS AND BURNS $29 OF ELECTRICITY A MONTH.' },
        ],
        alt: `Yes, but when the internet goes down I can still view my own router backups at 10 gigabits per second.`,
      },
      {
        cues: ['container', 'image', '18gb', 'bloated', 'layer'],
        layout: 'three_panel',
        prop: 'server_rack',
        cast: ['cueball', 'ponytail'],
        poses: [['hands_on_head', 'conversing'], ['pointing', 'chin_think'], ['shrug', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: `WHY IS OUR ${/\bdocker\b/i.test(sem.clean) ? 'DOCKER IMAGE' : np} ${sem.quantCue || '18GB'}?! THE BINARY IS TWO MEGABYTES!` },
          { speaker: 'right', text: 'LAYER 4 INSTALLED TEXLIVE, CHROMIUM, AND ALL OF UBUNTU FOR ONE FONT.' },
          { speaker: 'left', text: 'DO NOT TOUCH THE DOCKERFILE. IT TOOK THREE WEEKS TO LINK LIBSSL.' },
        ],
        alt: `Multi-stage builds were invented so you can leave the 17.9 GB crime scene in stage zero and never speak of it.`,
      },
    ];
  }

  if (domain === 'code') {
    return [
      {
        cues: ['css', 'div', 'centering', 'footer', 'safari', 'mobile'],
        layout: 'three_panel',
        prop: 'laptop_desk',
        cast: ['cueball', 'megan'],
        poses: [['typing', 'conversing'], ['hands_on_head', 'pointing'], ['shrug', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: `I MOVED ONE BUTTON TWO PIXELS LEFT IN CSS.` },
          { speaker: 'right', text: 'NOW THE FOOTER ON MOBILE SAFARI IS FLOATING OVER GREENLAND.' },
          { speaker: 'left', text: 'IF I ADD Z-INDEX: 999999 WE CAN PRETEND IT WAS ART DIRECTION.' },
        ],
        alt: `Every stylesheet eventually evolves two load-bearing '!important' rules locked in eternal stalemate.`,
      },
      {
        cues: ['regex', 'regular expression', 'hieroglyphs', 'yesterday'],
        layout: 'two_panel',
        prop: 'whiteboard',
        cast: ['ponytail', 'cueball'],
        poses: [['pointing', 'chin_think'], ['arms_crossed', 'hands_on_head']],
        turns: [
          { speaker: 'left', text: `YESTERDAY I WROTE THIS CLEAN REGEX TO PARSE ${ctx || 'INPUT STRINGS'}.` },
          { speaker: 'right', text: 'TODAY IT READS LIKE AN ANCIENT CURSE AND MATCHES CAT KEYBOARD WALKS.' },
        ],
        alt: `Regex is the only write-only notation where adding one question mark changes greediness and your weekend plans.`,
      },
      {
        cues: ['naming', 'boolean', 'variable', 'forty', 'minutes'],
        layout: 'three_panel',
        prop: 'laptop_desk',
        cast: ['cueball', 'ponytail'],
        poses: [['chin_think', 'conversing'], ['typing', 'pointing'], ['shrug', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: 'THE ALGORITHM TOOK TEN MINUTES. NAMING THIS BOOLEAN TOOK FORTY.' },
          { speaker: 'right', text: 'WHAT DID YOU PICK?' },
          { speaker: 'left', text: '`isNotUnreadyFlagV2Final`.' },
        ],
        alt: `Six months later someone negates it with '!' in an if-statement and nobody ever touches the module again.`,
      },
      {
        cues: ['rust', 'borrow', 'checker', 'linked', 'list'],
        layout: 'three_panel',
        prop: 'laptop_desk',
        cast: ['cueball', 'black_hat'],
        poses: [['typing', 'conversing'], ['hands_on_head', 'arms_crossed'], ['shrug', 'pointing']],
        turns: [
          { speaker: 'left', text: `THE BORROW CHECKER REJECTED MY ${ctx || 'LINKED LIST'} FOR THE NINTH TIME.` },
          { speaker: 'right', text: 'HAVE YOU TRIED CLONING EVERYTHING AND CALLING IT PROTOTYPE ARCHITECTURE?' },
          { speaker: 'left', text: 'THE COMPILER IS NOT MAD, JUST VERY DISAPPOINTED IN MY LIFETIMES.' },
        ],
        alt: `Once a Rust program finally compiles, you are legally allowed to frame the binary and take Thursday off.`,
      },
      {
        cues: ['dead', '2016', 'deleting', 'billing', 'unused'],
        layout: 'two_panel',
        prop: 'server_rack',
        cast: ['cueball', 'ponytail'],
        poses: [['pointing', 'conversing'], ['hands_on_head', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: `ZERO CALLERS REFERENCED ${np} SINCE 2016, SO I DELETED IT.` },
          { speaker: 'right', text: 'COOL, REFLECTION IN THE BILLING CRON JOB JUST PANICKED IN LATIN.' },
        ],
        alt: `Chesterton's Fence applies doubly to functions named 'doNotDelete_tempFix_v1_old'.`,
      },
      {
        cues: ['typescript', 'generic', 'scrolling', 'pages', 'error'],
        layout: 'two_panel',
        prop: 'laptop_desk',
        cast: ['megan', 'cueball'],
        poses: [['typing', 'pointing'], ['arms_crossed', 'hands_on_head']],
        turns: [
          { speaker: 'left', text: `OUR TYPE SYSTEM PROVES ${np} CANNOT FAIL AT RUNTIME!` },
          { speaker: 'right', text: 'THE ERROR MESSAGE IS SIX PAGES LONG AND ENDS WITH "STRING IS NOT NEVER".' },
        ],
        alt: `Somewhere on line 184 of the nested generic tooltip lies the missing comma you actually forgot.`,
      },
      {
        cues: ['print', 'debugger', 'crashed', 'console.log'],
        layout: 'three_panel',
        prop: 'laptop_desk',
        cast: ['cueball', 'ponytail'],
        poses: [['typing', 'conversing'], ['typing', 'pointing'], ['shrug', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: 'THE SYMBOLIC DEBUGGER CRASHED, SO WE FALL BACK TO THE ANCIENT WAYS.' },
          { speaker: 'right', text: 'PRINT("HERE 1"), PRINT("HERE 2"), PRINT("WHY GOD WHY")?' },
          { speaker: 'left', text: 'IT PRINTED "HERE 1" AND "HERE 3" WITHOUT "HERE 2". CALL A PRIEST.' },
        ],
        alt: `Removing the debug print statement makes the race condition disappear, so 'PRINT("DO NOT REMOVE")' ships to prod.`,
      },
      {
        cues: ['sleep(100)', 'comment', 'circumstances', 'remove'],
        layout: 'single_panel',
        prop: 'whiteboard',
        cast: ['ponytail', 'cueball'],
        turns: [
          { speaker: 'left', text: 'WHY IS THERE A HARDCODED SLEEP(100) WITH THREE SKULL EMOJIS ABOVE IT?' },
          { speaker: 'right', text: 'IT LETS THE DATABASE WIN A RACE CONDITION NOBODY UNDERSTANDS.' },
          { speaker: 'left', text: 'LEAVE IT. ADD A FOURTH SKULL.' },
        ],
        alt: `Archaeologists in 3026 will excavate our repo and recognize load-bearing timing delays as sacred ritual pauses.`,
      },
      {
        cues: ['pull', 'request', 'reviewing', '4200', 'lgtm'],
        layout: 'four_panel',
        prop: 'laptop_desk',
        cast: ['ponytail', 'cueball'],
        poses: [['typing', 'conversing'], ['pointing', 'hands_on_head'], ['typing', 'chin_think'], ['arms_crossed', 'shrug']],
        turns: [
          { speaker: 'left', text: 'TEN-LINE DIFF: 28 NITPICKS ABOUT VARIABLE RHYTHM.' },
          { speaker: 'right', text: `NOW REVIEW THIS ${sem.quantCue || '4,200-LINE'} ${/\btypescript\b/i.test(sem.clean) ? 'TYPESCRIPT PR' : 'PR'} BEFORE STANDUP.` },
          { speaker: 'left', text: 'SCROLL... SCROLL... SCROLL...' },
          { speaker: 'left', text: '"LOOKS GOOD TO ME! SHIP IT."' },
        ],
        alt: `Code review scrutiny is inversely proportional to diff size squared, dropping to zero right around line 400.`,
      },
    ];
  }

  if (domain === 'ops') {
    return [
      {
        cues: ['4:55pm', 'friday', 'one-line', 'tweak', 'config'],
        layout: 'two_panel',
        prop: 'laptop_desk',
        cast: ['cueball', 'ponytail'],
        poses: [['typing', 'conversing'], ['hands_on_head', 'pointing']],
        turns: [
          { speaker: 'left', text: `4:55 PM FRIDAY: JUST A HARMLESS ONE-LINE ${head} TWEAK.` },
          { speaker: 'right', text: '9:14 PM FRIDAY: LEARNING HOW BGP ROUTING WORKS BY FLASHLIGHT.' },
        ],
        alt: `The phrase 'it is only a config change, not real code' precedes 78% of all major industry post-mortems.`,
      },
      {
        cues: ['rebase', 'conflict', '90', 'clone', 'fresh'],
        layout: 'three_panel',
        prop: 'laptop_desk',
        cast: ['cueball', 'megan'],
        poses: [['typing', 'conversing'], ['hands_on_head', 'pointing'], ['shrug', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: `GIT SAYS I AM ON DETACHED HEAD REBASING ${sem.quantCue || '90 FILES'}.` },
          { speaker: 'right', text: 'USE REFLOG!' },
          { speaker: 'left', text: 'TOO LATE, I COPIED MY THREE CHANGED FILES TO DESKTOP AND RE-CLONED.' },
        ],
        alt: `Real git mastery is knowing the exact moment 'rm -rf repo && git clone' has higher expected value than '--continue'.`,
      },
      {
        cues: ['rebase', 'before', 'release'],
        layout: 'single_panel',
        prop: 'laptop_desk',
        cast: ['cueball', 'ponytail'],
        turns: [
          { speaker: 'left', text: 'JUST ONE CLEAN INTERACTIVE GIT REBASE RIGHT BEFORE RELEASE.' },
          { speaker: 'right', text: 'WHY DOES COMMIT 1 OF 47 SAY "REAPPLYING YOUR PAST SINS IN CHRONOLOGICAL ORDER"?' },
        ],
        alt: `Interactive rebase five minutes before the release cut is how two clean branches become one archaeological dig.`,
      },
      {
        cues: ['ssl', 'certificate', 'expiring', 'leap', 'sunday', 'midnight'],
        layout: 'two_panel',
        prop: 'server_rack',
        cast: ['ponytail', 'cueball'],
        poses: [['pointing', 'hands_on_head'], ['arms_crossed', 'shrug']],
        turns: [
          { speaker: 'left', text: `AUTO-RENEWAL WORKED IN STAGING! WHY DID ${np} EXPIRE AT MIDNIGHT?` },
          { speaker: 'right', text: 'THE CRON JOB BELONGED TO AN ENGINEER WHO LEFT IN 2022.' },
        ],
        alt: `Nothing unites an organization faster than a browser full-screen red triangle on Monday morning.`,
      },
      {
        cues: ['dns', 'network', 'six', 'hours', 'realizing'],
        layout: 'three_panel',
        prop: 'server_rack',
        cast: ['megan', 'cueball'],
        poses: [['conversing', 'typing'], ['pointing', 'chin_think'], ['arms_crossed', 'hands_on_head']],
        turns: [
          { speaker: 'left', text: 'IT CANNOT BE DNS. WE CHECKED DNS AT NOON.' },
          { speaker: 'right', text: 'WE TRACED PACKETS FOR SIX HOURS. WHAT WAS IT?' },
          { speaker: 'left', text: 'CACHED DNS.' },
        ],
        alt: `Even when the root cause is a severed undersea fiber cable, somehow a stale TTL makes it twice as confusing.`,
      },
      {
        cues: ['commit', 'history', 'fix2', 'asdfgh', 'messages'],
        layout: 'single_panel',
        prop: 'laptop_desk',
        cast: ['ponytail', 'cueball'],
        turns: [
          { speaker: 'left', text: '9 AM COMMIT: "FEAT: IMPLEMENT IDEMPOTENT RETRY PIPELINE."' },
          { speaker: 'left', text: '4 PM COMMIT: "FIX CI", "FIX CI FOR REAL", "PLEASE WORK", "ASDFGH".' },
          { speaker: 'right', text: 'SQUASH AND MERGE HIDES ALL HUMAN SUFFERING.' },
        ],
        alt: `Git blame on 'asdfgh' always points to yourself during an undocumented battle with CI environment variables.`,
      },
      {
        cues: ['status', 'page', 'green', 'icons', 'fire'],
        layout: 'two_panel',
        prop: 'whiteboard',
        cast: ['cueball', 'black_hat'],
        poses: [['pointing', 'arms_crossed'], ['hands_on_head', 'pointing']],
        turns: [
          { speaker: 'left', text: 'PRODUCTION IS ON FIRE, YET OUR PUBLIC STATUS PAGE IS ALL GREEN CHECKMARKS!' },
          { speaker: 'right', text: 'YEAH, THE ALERTING PIPELINE DEPENDS ON THE DATABASE THAT JUST CAUGHT FIRE.' },
        ],
        alt: `'Minor elevated latency for a subset of requests' is corporate translator speak for 'the rack melted'.`,
      },
      {
        cues: ['pagerduty', 'waking', 'test', 'health', 'alert'],
        layout: 'two_panel',
        prop: 'couch',
        cast: ['cueball', 'megan'],
        poses: [['hands_on_head', 'conversing'], ['shrug', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: `PAGERDUTY WOKE ME AT 3:18 AM FOR ${ctx || 'A STAGING HEALTH CHECK'}.` },
          { speaker: 'right', text: 'THE SEVERITY WAS P1 BECAUSE THE PAGER TEST SCRIPT TIMED OUT.' },
        ],
        alt: `Hearing the default PagerDuty ringtone in a grocery store years later still triggers an immediate adrenaline spike.`,
      },
      {
        cues: ['force', 'airplane', 'mode', 'main', 'plane'],
        layout: 'two_panel',
        prop: 'laptop_desk',
        cast: ['black_hat', 'cueball'],
        poses: [['typing', 'hands_on_head'], ['arms_crossed', 'pointing']],
        turns: [
          { speaker: 'left', text: 'FORCE-PUSHED TO MAIN. CABIN DOORS ARE CLOSING. SEE YOU IN TWELVE HOURS!' },
          { speaker: 'right', text: 'YOU ARE ON EVERY SECURITY NO-FLY LIST I CAN PERSONALLY INVENT.' },
        ],
        alt: `In-flight Wi-Fi costs $19, which is a bargain compared to watching 40 Slack notifications land simultaneously at baggage claim.`,
      },
    ];
  }

  if (domain === 'science') {
    return [
      {
        cues: ['orbital', 'kerbal', 'napkin', 'rocket', 'orbit'],
        layout: 'three_panel',
        prop: 'whiteboard',
        cast: ['cueball', 'megan'],
        poses: [['pointing', 'conversing'], ['conversing', 'pointing'], ['shrug', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: 'TO CATCH AN OBJECT AHEAD OF YOU IN ORBIT, YOU THRUST BACKWARD TO DROP LOWER!' },
          { speaker: 'right', text: 'DID YOU LEARN ASTRODYNAMICS AT MIT?' },
          { speaker: 'left', text: 'NO, STRANDING LITTLE GREEN ASTRONAUTS AROUND MUN FOR 600 HOURS.' },
        ],
        alt: `NASA mission planners quietly nod whenever someone explains Hohmann transfers using 'more boosters' units.`,
      },
      {
        cues: ['imperial', 'cups', 'ounces', 'relativistic', 'units'],
        layout: 'two_panel',
        prop: 'whiteboard',
        cast: ['ponytail', 'cueball'],
        poses: [['pointing', 'chin_think'], ['arms_crossed', 'shrug']],
        turns: [
          { speaker: 'left', text: 'AMERICAN RECIPES MEASURE FLOUR BY VOLUME, BUTTER BY STICKS, AND HEAT BY VIBES.' },
          { speaker: 'right', text: 'CONVERTING CUPS TO GRAMS REQUIRED DENSITY TABLES AND GENERAL RELATIVITY.' },
        ],
        alt: `One 'packed cup of brown sugar' varies by 38% depending on how stressful your afternoon was.`,
      },
      {
        cues: ['soup', 'blowing', 'stirring', 'thermodynamics', 'hot'],
        layout: 'three_panel',
        prop: 'pedestal',
        cast: ['cueball', 'megan'],
        poses: [['pointing', 'conversing'], ['chin_think', 'pointing'], ['hands_on_head', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: `EVAPORATIVE COOLING FROM BLOWING ON SOUP BEATS CONVECTIVE STIRRING BY 2.4X!` },
          { speaker: 'right', text: 'SO IT IS SAFE TO EAT?' },
          { speaker: 'left', text: 'ONLY THE TOP MILLIMETER. BELOW THAT IS STILL LIQUID MAGMA.' },
        ],
        alt: `Mouth tissue thermal conductivity research is funded almost entirely by impatient people eating tomato bisque.`,
      },
      {
        cues: ['proof', 'intimidation', 'linear', 'algebra', 'lecture', 'trivial'],
        layout: 'two_panel',
        prop: 'pointer_stick',
        cast: ['hairbun', 'cueball'],
        poses: [['pointer_rod', 'chin_think'], ['arms_crossed', 'hands_on_head']],
        turns: [
          { speaker: 'left', text: 'FROM EQUATION 2 IT TRIVIALLY FOLLOWS THAT THE EIGENVALUES VANISH.' },
          { speaker: 'right', text: '"TRIVIALLY" JUST SKIPPED FOURTEEN PAGES OF GRADUATE ALGEBRA.' },
        ],
        alt: `Whenever a textbook says 'left as an easy exercise for the reader', the author lost their scratch paper in 1984.`,
      },
      {
        cues: ['clocks', 'moon', 'earth', 'daylight', 'synchronize'],
        layout: 'single_panel',
        prop: 'whiteboard',
        cast: ['ponytail', 'cueball', 'black_hat'],
        turns: [
          { speaker: 'left', text: 'LUNAR CLOCKS GAIN 56 MICROSECONDS PER EARTH DAY FROM LOWER GRAVITY.' },
          { speaker: 'right', text: 'WE CAN HANDLE GENERAL RELATIVITY!' },
          { speaker: 'third', text: 'NOW ADD ARIZONA AND DAYLIGHT SAVING TIME.' },
        ],
        alt: `Einstein solved curved spacetime in ten years and wisely never attempted calendar invite recurrence rules.`,
      },
      {
        cues: ['bayesian', 'keys', 'fridge', 'lost', 'prior'],
        layout: 'three_panel',
        prop: 'whiteboard',
        cast: ['cueball', 'megan'],
        poses: [['pointing', 'conversing'], ['conversing', 'pointing'], ['hands_on_head', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: 'MY BAYESIAN PRIOR THAT MY KEYS ARE IN THE FRIDGE IS 0.0001.' },
          { speaker: 'right', text: 'YOU ALREADY CHECKED EVERY NORMAL SURFACE TWICE.' },
          { speaker: 'left', text: 'POSTERIOR UPDATED. THEY ARE BEHIND THE OAT MILK.' },
        ],
        alt: `Holding cold groceries in one hand and keys in the other creates a brief neurological swap operation.`,
      },
      {
        cues: ['physicist', 'horse', 'frictionless', 'cylinder', 'sphere'],
        layout: 'two_panel',
        prop: 'whiteboard',
        cast: ['ponytail', 'beret_guy'],
        poses: [['pointing', 'conversing'], ['arms_crossed', 'shrug']],
        turns: [
          { speaker: 'left', text: `IF WE APPROXIMATE ${np} AS A UNIFORM CYLINDER IN A VACUUM, IT WORKS!` },
          { speaker: 'right', text: 'THE RANCHERS ASKED US TO STOP PUTTING THEIR HORSES IN VACUUM CHAMBERS.' },
        ],
        alt: `Dropping higher-order air-resistance terms works great until the cylinder tries to eat your lab notebook.`,
      },
      {
        cues: ['exoplanet', 'astronomers', 'naming', 'hd-40291b'],
        layout: 'two_panel',
        prop: 'pointer_stick',
        cast: ['megan', 'cueball'],
        poses: [['pointer_rod', 'conversing'], ['pointing', 'shrug']],
        turns: [
          { speaker: 'left', text: 'THIS WORLD RAINS MOLTEN GLASS SIDEWAYS AT 5,000 MPH.' },
          { speaker: 'right', text: 'AND ASTRONOMERS NAMED IT SOMETHING LIKE "GLIESE-402B" INSTEAD OF "DOOMSPHÈRE"?' },
        ],
        alt: `IAU naming rules exist specifically to stop grad students from naming hot Jupiters 'Spicy Bois-7'.`,
      },
      {
        cues: ['cumulus', 'clouds', 'dew'],
        layout: 'two_panel',
        prop: 'whiteboard',
        cast: ['ponytail', 'cueball'],
        poses: [['pointing', 'conversing'], ['chin_think', 'pointing']],
        turns: [
          { speaker: 'left', text: 'CUMULUS CLOUDS ARE FLAT ON THE BOTTOM BECAUSE RISING AIR HITS THE DEW-POINT ALTITUDE!' },
          { speaker: 'right', text: 'SO WE CAN LITERALLY SEE AN INVISIBLE THERMODYNAMIC PHASE BOUNDARY FLOATING OVERHEAD.' },
        ],
        alt: `Once someone points out the lifting condensation level, every summer sky looks like an atmospheric graph.`,
      },
    ];
  }

  return null;
}

function buildRemainingBlueprints(domain, sem) {
  const np = up(sem.topicNP);
  const npLow = low(sem.topicNP);
  const head = up(sem.singularHead);
  const ctx = sem.contextTag ? up(sem.contextTag) : '';

  if (domain === 'security') {
    return [
      {
        cues: ['password', 'policy', 'hieroglyph', 'prime', 'fear'],
        layout: 'three_panel',
        prop: 'laptop_desk',
        cast: ['cueball', 'megan'],
        poses: [['typing', 'conversing'], ['hands_on_head', 'pointing'], ['shrug', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: 'PASSWORD MUST INCLUDE A PRIME, AN ELEMENT SYMBOL, AND A CHILDHOOD REGRET.' },
          { speaker: 'right', text: 'AND IT EXPIRES EVERY THIRTY DAYS?' },
          { speaker: 'left', text: 'YES, SO EVERYONE WRITES IT ON A YELLOW STICKY NOTE UNDER THE KEYBOARD.' },
        ],
        alt: `Next month's rotation simply increments the trailing exclamation mark from '!' to '!!'.`,
      },
      {
        cues: ['two-factor', '2fa', 'phone', 'room', 'couch'],
        layout: 'two_panel',
        prop: 'couch',
        cast: ['megan', 'cueball'],
        poses: [['conversing', 'pointing'], ['shrug', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: 'ENTER THE SIX-DIGIT CODE SENT TO YOUR PHONE ON THE OTHER SIDE OF THE ROOM.' },
          { speaker: 'right', text: 'GUESS I DID NOT NEED TO PAY TAXES OR ACCESS MY BANK ACCOUNT THIS YEAR.' },
        ],
        alt: `If you do stand up and walk over, the 30-second TOTP timer is contractually obligated to hit 01 on arrival.`,
      },
      {
        cues: ['cookie', 'consent', 'banner', '340', 'legitimate', 'partners'],
        layout: 'three_panel',
        prop: 'laptop_desk',
        cast: ['cueball', 'black_hat'],
        poses: [['typing', 'conversing'], ['pointing', 'arms_crossed'], ['hands_on_head', 'shrug']],
        turns: [
          { speaker: 'left', text: 'I JUST WANTED TO CHECK TODAYS WEATHER FORECAST.' },
          { speaker: 'right', text: 'PLEASE TOGGLE 340 AD PARTNERS CLAIMING "LEGITIMATE INTEREST" INDIVIDUALLY.' },
          { speaker: 'left', text: 'ILL JUST LOOK OUT THE WINDOW.', reply: 'THE WINDOW ALSO SETS A TRACKER.' },
        ],
        alt: `Clicking 'Manage Preferences' opens a scroll bar thumb smaller than a single hydrogen atom.`,
      },
      {
        cues: ['ssh', 'keys', 'rotating', 'locking', 'jump', 'box'],
        layout: 'two_panel',
        prop: 'server_rack',
        cast: ['ponytail', 'cueball'],
        poses: [['pointing', 'hands_on_head'], ['arms_crossed', 'shrug']],
        turns: [
          { speaker: 'left', text: 'GOOD NEWS: ZERO UNAUTHORIZED HUMANS CAN ACCESS THE PRODUCTION JUMP BOX.' },
          { speaker: 'right', text: 'BAD NEWS: ZERO AUTHORIZED HUMANS CAN ACCESS IT EITHER.' },
        ],
        alt: `Mathematically, bricking sshd achieves 100% confidentiality and 100% integrity at the small cost of 0% availability.`,
      },
      {
        cues: ['phishing', 'training', 'hr', 'legitimate', 'email'],
        layout: 'three_panel',
        prop: 'laptop_desk',
        cast: ['megan', 'ponytail'],
        poses: [['typing', 'conversing'], ['pointing', 'chin_think'], ['arms_crossed', 'shrug']],
        turns: [
          { speaker: 'left', text: 'I REPORTED THAT SKETCHY EXTERNAL LINK TO SECURITY IMMEDIATELY!' },
          { speaker: 'right', text: 'THAT WAS OUR ACTUAL HEALTH BENEFITS ENROLLMENT PORTAL.' },
          { speaker: 'left', text: 'WHY DOES REAL HR USE FOUR MISSPELLED SUBDOMAINS AND COMIC SANS?' },
        ],
        alt: `Meanwhile the simulated phishing test had valid DKIM, zero typos, and a cleaner UI than payroll.`,
      },
      {
        cues: ['encrypting', 'grocery', 'milk', 'limes', 'end-to-end'],
        layout: 'two_panel',
        prop: 'laptop_desk',
        cast: ['cueball', 'megan'],
        poses: [['typing', 'conversing'], ['shrug', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: `OUR ${np} NOW USES POST-QUANTUM LATTICE ENCRYPTION!` },
          { speaker: 'right', text: 'THE NSA WILL NEVER LEARN THAT WE ARE OUT OF LIMES AND PAPER TOWELS.' },
        ],
        alt: `Key exchange took twenty minutes, so I bought the wrong tortilla chips in plaintext anyway.`,
      },
      {
        cues: ['audit', 'taped', 'monitor', 'admin'],
        layout: 'two_panel',
        prop: 'whiteboard',
        cast: ['hairbun', 'beret_guy'],
        poses: [['pointing', 'conversing'], ['hands_on_head', 'shrug']],
        turns: [
          { speaker: 'left', text: 'WE PASSED SOC-2 WITH ZERO CRITICAL FINDINGS!' },
          { speaker: 'right', text: 'AWESOME, LET ME TAKE THE ADMIN PASSWORD OFF THE OFFICE WHITEBOARD.' },
        ],
        alt: `Security auditors only inspect digital PDFs, making a physical Post-it note technically air-gapped.`,
      },
      {
        cues: ['smartfridge', 'fridge', 'botnet', 'ice', 'maker'],
        layout: 'two_panel',
        prop: 'pedestal',
        cast: ['cueball', 'ponytail'],
        poses: [['pointing', 'conversing'], ['hands_on_head', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: 'WHY IS OUR ROUTER SENDING 40 GIGABITS OF TRAFFIC TO EASTERN EUROPE?' },
          { speaker: 'right', text: 'NOBODY PATCHED THE ICE MAKER AND NOW THE FREEZER MINES MONERO.' },
        ],
        alt: `Unplugging Ethernet from the crisper drawer disabled crushed ice, as the founders of TCP/IP intended.`,
      },
    ];
  }

  if (domain === 'work') {
    return [
      {
        cues: ['pre-meeting', 'align', 'alignment', 'agenda', 'sync'],
        layout: 'three_panel',
        prop: 'whiteboard',
        cast: ['ponytail', 'cueball'],
        poses: [['pointing', 'conversing'], ['conversing', 'chin_think'], ['arms_crossed', 'hands_on_head']],
        turns: [
          { speaker: 'left', text: 'WELCOME TO TUESDAY PRE-SYNC TO ALIGN ON FRIDAY ALIGNMENT SYNC.' },
          { speaker: 'right', text: 'WHEN DO WE ACTUALLY BUILD THE THING?' },
          { speaker: 'left', text: 'LET US SCHEDULE A SCOPING WORKSHOP TO EXPLORE THAT QUESTION.' },
        ],
        alt: `By Q3 the alignment meetings achieved self-sustaining criticality and no longer required a product.`,
      },
      {
        cues: ['jira', 'ticket', 'low', 'priority', 'obama', 'backlog'],
        layout: 'two_panel',
        prop: 'laptop_desk',
        cast: ['cueball', 'megan'],
        poses: [['typing', 'conversing'], ['shrug', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: 'THIS BACKLOG TICKET JUST TURNED OLD ENOUGH TO LEARN FRACTIONS.' },
          { speaker: 'right', text: 'DO NOT CLOSE IT! THREE MANAGERS USE IT AS EMOTIONAL INFRASTRUCTURE.' },
        ],
        alt: `Every six months a bot comments 'still relevant?' and someone replies 'yes, moving to next sprint' until the sun dies.`,
      },
      {
        cues: ['offline', 'take', 'speaking', 'never'],
        layout: 'two_panel',
        prop: 'whiteboard',
        cast: ['hairbun', 'cueball'],
        poses: [['pointing', 'conversing'], ['arms_crossed', 'shrug']],
        turns: [
          { speaker: 'left', text: '"GREAT POINT — LET US TAKE THIS OFFLINE."' },
          { speaker: 'right', text: 'TRANSLATION: CAST THIS TOPIC INTO THE SEA AND NEVER SPEAK OF IT AGAIN.' },
        ],
        alt: `Somewhere there is a mythical 'Offline' dimension filled with 90,000 unresolved architecture debates.`,
      },
      {
        cues: ['sharing', 'screen', '190', 'chaotic', 'tabs', 'revealing'],
        layout: 'three_panel',
        prop: 'laptop_desk',
        cast: ['cueball', 'ponytail'],
        poses: [['typing', 'conversing'], ['hands_on_head', 'pointing'], ['shrug', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: 'LET ME SHARE MY DESKTOP REAL QUICK.' },
          { speaker: 'right', text: 'WE CAN ALL SEE 190 TABS AND A SEARCH FOR "HOW TO BOIL AN EGG".' },
          { speaker: 'left', text: 'IT WAS FOR BENCHMARKING THERMAL TRANSFER!' },
        ],
        alt: `Sharing 'Entire Screen' instead of 'Single Window' is an extreme sport with zero upside.`,
      },
      {
        cues: ['spreadsheet', 'finance', 'vlookup', 'vlookups', 'excel'],
        layout: 'two_panel',
        prop: 'laptop_desk',
        cast: ['ponytail', 'cueball'],
        poses: [['pointing', 'chin_think'], ['arms_crossed', 'hands_on_head']],
        turns: [
          { speaker: 'left', text: 'OUR ENTIRE $80M SUPPLY CHAIN RUNS ON `FINAL_V7_GREG_COPY2.XLSX`.' },
          { speaker: 'right', text: 'CELL D14 HAS NINE NESTED VLOOKUPS AND GREG RETIRED IN 2020.' },
        ],
        alt: `Every Fortune 500 cloud migration eventually discovers one load-bearing Excel macro running under a desk.`,
      },
      {
        cues: ['calendar', '15-minute', 'gap', 'focus'],
        layout: 'two_panel',
        prop: 'laptop_desk',
        cast: ['megan', 'cueball'],
        poses: [['typing', 'conversing'], ['hands_on_head', 'pointing']],
        turns: [
          { speaker: 'left', text: 'YOU HAVE ONE FIFTEEN-MINUTE OPEN BLOCK BETWEEN 9 AM AND 5 PM.' },
          { speaker: 'right', text: 'SOMEONE JUST BOOKED IT FOR A "QUICK PICK YOUR BRAIN" CHAT.' },
        ],
        alt: `Fourteen minutes isn't even enough time to remember what programming language this repo uses.`,
      },
      {
        cues: ['unmuting', '60-person', 'video', 'sorry', 'ahead'],
        layout: 'single_panel',
        prop: 'laptop_desk',
        cast: ['cueball', 'megan'],
        turns: [
          { speaker: 'left', text: 'NO, SORRY — YOU GO AHEAD.' },
          { speaker: 'right', text: 'NO, PLEASE, GO AHEAD!' },
          { speaker: 'left', text: 'TEN SECONDS OF DEAD SILENCE FOLLOWED BY BOTH SPEAKING AT ONCE AGAIN.' },
        ],
        alt: `Ethernet solved packet collision backoff in 1980, yet human video calls still lack exponential jitter.`,
      },
      {
        cues: ['roadmap', 'q4', 'slide', 'magic'],
        layout: 'two_panel',
        prop: 'pointer_stick',
        cast: ['hairbun', 'beret_guy'],
        poses: [['pointer_rod', 'conversing'], ['arms_crossed', 'shrug']],
        turns: [
          { speaker: 'left', text: 'Q1 IS RESEARCH, Q2 IS PROTOTYPE, AND Q4 IS 10X GLOBAL REVENUE.' },
          { speaker: 'right', text: 'WHAT HAPPENS IN Q3?', reply: 'THE RECTANGLE JUST SAYS "SYNERGY".' },
        ],
        alt: `Drawing the arrow thicker in PowerPoint increases engineering velocity by at least 0%.`,
      },
    ];
  }

  if (domain === 'food') {
    return [
      {
        cues: ['avocado', 'ripeness', '11', 'window', 'tuesday'],
        layout: 'three_panel',
        prop: 'pedestal',
        cast: ['megan', 'cueball'],
        poses: [['pointing', 'conversing'], ['chin_think', 'conversing'], ['hands_on_head', 'shrug']],
        turns: [
          { speaker: 'left', text: 'NOT YET... NOT YET... STILL SOLID GRANITE...' },
          { speaker: 'right', text: 'YOU BLINKED FOR FOUR SECONDS.' },
          { speaker: 'left', text: 'MISSED IT. IT IS NOW BROWN COMPOST.' },
        ],
        alt: `Avocados Ripen solely during the eleven minutes you are in another zip code.`,
      },
      {
        cues: ['kale', 'meal-prep', 'sunday', 'optimism', 'spinach'],
        layout: 'two_panel',
        prop: 'couch',
        cast: ['cueball', 'megan'],
        poses: [['conversing', 'pointing'], ['shrug', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: 'ON SUNDAY I BOUGHT TWO POUNDS OF ORGANIC KALE FOR HEALTHY SALADS.' },
          { speaker: 'right', text: 'BY THURSDAY YOU ORDERED PAD THAI WHILE THE KALE LIQUEFIED IN PEACE.' },
        ],
        alt: `The refrigerator crisper drawer is an expensive witness-protection program for vegetables before compost.`,
      },
      {
        cues: ['sourdough', 'starter', 'houseplant', 'care'],
        layout: 'three_panel',
        prop: 'pedestal',
        cast: ['beret_guy', 'ponytail'],
        poses: [['pointing', 'conversing'], ['conversing', 'arms_crossed'], ['shrug', 'pointing']],
        turns: [
          { speaker: 'left', text: 'I NAMED MY SOURDOUGH STARTER GREGORY AND FEED HIM ARTISAN RYE AT 7 AM.' },
          { speaker: 'right', text: 'HOW MANY LOAVES HAVE YOU BAKED?' },
          { speaker: 'left', text: 'ZERO, I AM NOT EMOTIONALLY READY TO BAKE MY SON.' },
        ],
        alt: `Discard pancakes are just tribute offerings to keep the jar from overflowing onto the counter.`,
      },
      {
        cues: ['pasta', 'boiling', 'box', '9', '11', 'trash'],
        layout: 'three_panel',
        prop: 'pedestal',
        cast: ['cueball', 'megan'],
        poses: [['conversing', 'pointing'], ['hands_on_head', 'conversing'], ['shrug', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: 'POURED THE PASTA IN AND THREW THE CARDBOARD BOX IN THE TRASH.' },
          { speaker: 'right', text: 'HOW MANY MINUTES DO WE BOIL IT?' },
          { speaker: 'left', text: 'DIGGING THE BOX BACK OUT OF THE TRASH LIKE EVERY OTHER NIGHT.' },
        ],
        alt: `I have cooked penne 400 times and my brain still refuses to allocate 4 bytes of permanent storage for '10 min'.`,
      },
      {
        cues: ['spicy', 'thai', 'level', '4', 'hubris'],
        layout: 'two_panel',
        prop: 'couch',
        cast: ['cueball', 'ponytail'],
        poses: [['pointing', 'conversing'], ['hands_on_head', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: '7:05 PM: "LEVEL 4 SPICY PLEASE, I LIKE A REAL KICK."' },
          { speaker: 'right', text: '7:22 PM: SEEING THROUGH TIME AND DRINKING HALF A GALLON OF MILK.' },
        ],
        alt: `When the restaurant asks 'regular 4 or Thai 4?', choosing the second option signs a legal waiver with your stomach.`,
      },
      {
        cues: ['waffle', 'hotel', 'breakfast', 'iron', 'tolerances'],
        layout: 'two_panel',
        prop: 'pedestal',
        cast: ['beret_guy', 'cueball'],
        poses: [['pointing', 'conversing'], ['conversing', 'chin_think']],
        turns: [
          { speaker: 'left', text: 'THE HOTEL ROTATING WAFFLE IRON IS PEAK HUMAN MECHANICAL ENGINEERING.' },
          { speaker: 'right', text: 'ZERO TRAINING, ONE FLIP, BEEP AT 150 SECONDS, PERFECT HEXAGONAL GRID.' },
        ],
        alt: `Compare that to the hotel shower handle, which requires a degree in fluid dynamics to find lukewarm.`,
      },
      {
        cues: ['decaf', '9pm', 'ceiling', '2am', 'staring'],
        layout: 'two_panel',
        prop: 'couch',
        cast: ['megan', 'cueball'],
        poses: [['conversing', 'hands_on_head'], ['arms_crossed', 'shrug']],
        turns: [
          { speaker: 'left', text: 'IT WAS DECAF! IT ONLY HAD FOUR MILLIGRAMS OF CAFFEINE!' },
          { speaker: 'right', text: 'AFTER AGE THIRTY, EVEN LOOKING AT A COFFEE BEAN AFTER 4 PM CANCELS SLEEP.' },
        ],
        alt: `2:15 AM is when your brain helpfully replays every awkward handshake you executed in 2014.`,
      },
      {
        cues: ['espresso', 'grind', '7am', 'dialing'],
        layout: 'three_panel',
        prop: 'pedestal',
        cast: ['cueball', 'megan'],
        poses: [['pointing', 'conversing'], ['hands_on_head', 'pointing'], ['shrug', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: `DIALING IN ${np} ${ctx ? `BEFORE ${ctx}` : 'AT DAWN'} REQUIRES CAFFEINE FIRST.` },
          { speaker: 'right', text: 'THAT IS A CLASSIC BOOTSTRAP DEADLOCK.' },
          { speaker: 'left', text: 'I WEIGHED THE BEANS TO 0.1 GRAMS AND FORGOT TO PUT A MUG UNDER THE SPOUT.' },
        ],
        alt: `Nothing humbles precision kitchen engineering faster than brewing $22 single-origin coffee straight into the drip tray.`,
      },
      {
        cues: ['drawer', 'masher', 'potato', 'spatula', 'jammed', 'utensil'],
        layout: 'two_panel',
        prop: 'pedestal',
        cast: ['megan', 'cueball'],
        poses: [['hands_on_head', 'conversing'], ['pointing', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: 'THE KITCHEN DRAWER OPENS TWO INCHES AND LOCKS LIKE A BANK VAULT.' },
          { speaker: 'right', text: 'ONE SIDEWAYS POTATO MASHER HAS TAKEN THE ENTIRE COUNTERTOP HOSTAGE.' },
        ],
        alt: `You can either reach blind into the two-inch gap like a raccoon or accept you no longer own spoons.`,
      },
    ];
  }

  if (domain === 'pets') {
    return [
      {
        cues: ['cat', '$90', 'velvet', 'cardboard', 'box', 'amazon'],
        layout: 'two_panel',
        prop: 'couch',
        cast: ['megan', 'cueball'],
        poses: [['pointing', 'conversing'], ['shrug', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: 'WE BOUGHT THE CAT A $90 ORTHOPEDIC VELVET LOUNGER.' },
          { speaker: 'right', text: 'HE HAS LIVED EXCLUSIVELY INSIDE THE SHIPPING CARDBOARD BOX SINCE TUESDAY.' },
        ],
        alt: `If you throw away the velvet bed and keep the box, the cat immediately transfers loyalty to a crumpled grocery receipt.`,
      },
      {
        cues: ['houseplant', 'weeds', 'sidewalk', 'concrete', 'water', 'milliliters'],
        layout: 'two_panel',
        prop: 'pedestal',
        cast: ['cueball', 'ponytail'],
        poses: [['hands_on_head', 'conversing'], ['pointing', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: 'MY HOUSEPLANT DIED BECAUSE TAP WATER WAS TWO DEGREES TOO COOL.' },
          { speaker: 'right', text: 'MEANWHILE A DANDELION OUTSIDE JUST SPLIT FOUR INCHES OF ASPHALT IN JANUARY.' },
        ],
        alt: `Tropical ferns survived the Cretaceous asteroid impact and lose the will to live if you move their pot six inches left.`,
      },
      {
        cues: ['crow', 'puzzle', 'morning', 'brain'],
        layout: 'two_panel',
        prop: 'pedestal',
        cast: ['megan', 'beret_guy'],
        poses: [['pointing', 'conversing'], ['conversing', 'shrug']],
        turns: [
          { speaker: 'left', text: 'THIS CROW BENT A WIRE HOOK TO SOLVE A THREE-STAGE WATER PUZZLE.' },
          { speaker: 'right', text: 'THIS MORNING I TRIED TO UNLOCK MY FRONT DOOR WITH MY CAR KEY FOB.' },
        ],
        alt: `Give corvids thumbs and thirty years and they will be filing bug reports against our municipal trash cans.`,
      },
      {
        cues: ['3:14am', 'zoomies', 'hallway', 'sprinting'],
        layout: 'two_panel',
        prop: 'couch',
        cast: ['cueball', 'megan'],
        poses: [['hands_on_head', 'conversing'], ['shrug', 'pointing']],
        turns: [
          { speaker: 'left', text: 'WHY IS THE CAT SPRINTING AT MACH 2 DOWN THE HALLWAY AT 3:14 AM?' },
          { speaker: 'right', text: 'CONSERVATION OF MOMENTUM. HE SLEPT FOR NINETEEN HOURS STRAIGHT.' },
        ],
        alt: `At 3:15 AM he stares intensely at an empty corner of the ceiling so you never fall back asleep either.`,
      },
      {
        cues: ['dog', 'vacuum', 'eldritch', 'invader'],
        layout: 'two_panel',
        prop: 'couch',
        cast: ['beret_guy', 'megan'],
        poses: [['conversing', 'pointing'], ['shrug', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: `ACCORDING TO THE DOG, ${np} IS A ROARING INTERDIMENSIONAL BEAST.` },
          { speaker: 'right', text: 'YET HE BRAVELY PROTECTED THE COUCH BY BARKING FOR TWENTY MINUTES.' },
        ],
        alt: `Every single week the vacuum retreats back into the hall closet, proving the barking has a 100% success rate.`,
      },
      {
        cues: ['compound', 'interest', 'retriever', 'explaining', 'tennis'],
        layout: 'two_panel',
        prop: 'whiteboard',
        cast: ['cueball', 'beret_guy'],
        poses: [['pointing', 'conversing'], ['shrug', 'conversing']],
        turns: [
          { speaker: 'left', text: 'IF YOU SAVE ONE TENNIS BALL TODAY, IN TEN YEARS YOU GET TWO TENNIS BALLS!' },
          { speaker: 'right', text: 'COUNTERPROPOSAL: THROW THE BALL RIGHT NOW AND NEVER STOP.' },
        ],
        alt: `Dogs apply a hyperbolic discount rate of infinity per second, which still outperforms most hedge funds.`,
      },
      {
        cues: ['seagull', 'projectile', 'beach', 'picnic', 'trajectories'],
        layout: 'two_panel',
        prop: 'whiteboard',
        cast: ['ponytail', 'cueball'],
        poses: [['pointing', 'conversing'], ['hands_on_head', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: 'SEAGULLS COMPUTE WIND SHEAR AND INTERCEPT VECTORS IN 80 MILLISECONDS.' },
          { speaker: 'right', text: 'HE EXTRACTED MY FRENCH FRY MID-BITE WITHOUT TOUCHING MY FINGERS.' },
        ],
        alt: `Lockheed spends billions on guidance firmware that a beach gull runs on three tortilla crumbs.`,
      },
      {
        cues: ['mosquito', 'ankle', 'bug', 'spray', 'centimeters'],
        layout: 'two_panel',
        prop: 'whiteboard',
        cast: ['cueball', 'megan'],
        poses: [['pointing', 'conversing'], ['arms_crossed', 'shrug']],
        turns: [
          { speaker: 'left', text: 'I COVERED 99.4% OF MY SKIN SURFACE AREA IN DEET.' },
          { speaker: 'right', text: 'AND ONE MOSQUITO STILL MAPPED THE UNGUARDED PATCH ON YOUR LEFT ANKLE.' },
        ],
        alt: `Mosquitoes use infrared thermal QA testing more thorough than aerospace weld inspection.`,
      },
      {
        cues: ['squirrel', 'feeder', 'bird', '$120', 'squirrel-proof'],
        layout: 'three_panel',
        prop: 'pedestal',
        cast: ['cueball', 'ponytail'],
        poses: [['pointing', 'conversing'], ['conversing', 'chin_think'], ['hands_on_head', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: 'THIS $120 BIRD FEEDER HAS SPRING BAFFLES AND AEROSPACE BEARINGS!' },
          { speaker: 'right', text: 'THE SQUIRREL IS HANGING UPSIDE DOWN EATING SUNFLOWER SEEDS RIGHT NOW.' },
          { speaker: 'left', text: 'WE ARE UNWITTINGLY RUNNING AN OLYMPIC TRAINING FACILITY FOR RODENTS.' },
        ],
        alt: `Every 'squirrel-proof' upgrade selectively breeds backyard squirrels capable of Mission Impossible wire stunts.`,
      },
    ];
  }

  if (domain === 'life') {
    return [
      {
        cues: ['furniture', 'flat-pack', 'swedish', 'dowels', 'leftover'],
        layout: 'three_panel',
        prop: 'pedestal',
        cast: ['cueball', 'megan'],
        poses: [['pointing', 'conversing'], ['chin_think', 'pointing'], ['shrug', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: 'THE BOOKSHELF IS UP AND I IMPROVED EFFICIENCY BY THREE WOODEN DOWELS!' },
          { speaker: 'right', text: 'LEFTOVER STRUCTURAL FASTENERS ARE NOT AN EFFICIENCY GAIN.' },
          { speaker: 'left', text: 'AS LONG AS NOBODY PUTS BOOKS ON IT, IT IS ROCK SOLID.' },
        ],
        alt: `Step 14's tiny cartoon person frowning next to a crossed-out hammer always arrives two steps after you used the hammer.`,
      },
      {
        cues: ['taxes', 'filing', 'government', 'knows', 'guess'],
        layout: 'two_panel',
        prop: 'laptop_desk',
        cast: ['megan', 'cueball'],
        poses: [['typing', 'conversing'], ['hands_on_head', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: 'THE IRS ALREADY HAS MY W-2. CAN THEY TELL ME WHAT I OWE?' },
          { speaker: 'right', text: 'NO, YOU MUST SOLVE A 40-LINE WORKSHEET AND GO TO JAIL IF WE DISAGREE.' },
        ],
        alt: `Imagine a grocery store scanning your cart, hiding the screen, and fining you if you guess the total wrong.`,
      },
      {
        cues: ['dentist', 'instruments', 'mouth', 'philosophical', 'questions'],
        layout: 'two_panel',
        prop: 'couch',
        cast: ['ponytail', 'cueball'],
        poses: [['pointing', 'conversing'], ['hands_on_head', 'shrug']],
        turns: [
          { speaker: 'left', text: 'SO WHAT DO YOU THINK ABOUT MONETARY POLICY THIS YEAR? DO NOT MOVE.' },
          { speaker: 'right', text: 'AAH-UUHGH-MMMPH.' },
        ],
        alt: `Dental school includes two semesters on understanding vowel-only replies around a suction tube.`,
      },
      {
        cues: ['gym', 'canceling', 'membership', 'notarized', 'scroll'],
        layout: 'three_panel',
        prop: 'laptop_desk',
        cast: ['cueball', 'black_hat'],
        poses: [['typing', 'conversing'], ['pointing', 'arms_crossed'], ['hands_on_head', 'shrug']],
        turns: [
          { speaker: 'left', text: 'JOINING THE GYM TOOK ONE TAP ON MY PHONE.' },
          { speaker: 'right', text: 'TO CANCEL, MAIL A NOTARIZED WAX SEAL TO OUR PO BOX DURING AN ECLIPSE.' },
          { speaker: 'left', text: 'MOVING TO ANOTHER COUNTRY MIGHT HAVE FEWER FORMS.' },
        ],
        alt: `Technically canceling your credit card and changing your legal name is the supported self-serve flow.`,
      },
      {
        cues: ['shampoo', 'conditioner', 'bottles', 'out-of-phase'],
        layout: 'two_panel',
        prop: 'whiteboard',
        cast: ['megan', 'cueball'],
        poses: [['pointing', 'conversing'], ['chin_think', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: 'SHAMPOO AND CONDITIONER HAVE INCOMMENSURATE USAGE PERIODS.' },
          { speaker: 'right', text: 'BY LEAST COMMON MULTIPLE THEY REALIGN ONCE EVERY 14.6 YEARS.' },
        ],
        alt: `Same theorem governs hot dogs (packs of 10) and buns (packs of 8), keeping the condiment industry solvent.`,
      },
      {
        cues: ['parallel', 'parking', 'watches', 'geometry', 'alone'],
        layout: 'two_panel',
        prop: 'whiteboard',
        cast: ['cueball', 'ponytail'],
        poses: [['pointing', 'conversing'], ['hands_on_head', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: 'ALONE AT NIGHT I PARALLEL PARK IN ONE FLUID GEOMETRIC ARC.' },
          { speaker: 'right', text: 'ONE PEDESTRIAN GLANCES OVER AND SUDDENLY YOU ARE SEVEN FEET FROM THE CURB.' },
        ],
        alt: `Observer effect in quantum mechanics was first discovered watching someone try to back into a spot outside a cafe.`,
      },
      {
        cues: ['laundry', 'chair', 'geological', 'stratification'],
        layout: 'two_panel',
        prop: 'couch',
        cast: ['cueball', 'megan'],
        poses: [['pointing', 'conversing'], ['arms_crossed', 'shrug']],
        turns: [
          { speaker: 'left', text: 'TOO CLEAN FOR THE HAMPER, TOO WORN FOR THE CLOSET: ENTER THE CHAIR.' },
          { speaker: 'right', text: 'CARBON DATING SHOWS THE LOWER SWEATER LAYER IS FROM NOVEMBER.' },
        ],
        alt: `Moving the pile to the bed at night and back to the chair in the morning is a closed thermodynamic cycle.`,
      },
      {
        cues: ['alarm', '6:45', '6:50', '6:53', 'stack'],
        layout: 'two_panel',
        prop: 'couch',
        cast: ['megan', 'cueball'],
        poses: [['pointing', 'conversing'], ['hands_on_head', 'shrug']],
        turns: [
          { speaker: 'left', text: 'WHY DO YOU HAVE ALARMS FOR 6:45, 6:50, 6:53, 6:58, AND 7:02 AM?' },
          { speaker: 'right', text: 'NIGHT ME AND MORNING ME ARE LOCKED IN ASYMMETRIC WARFARE.' },
        ],
        alt: `Setting 6:53 instead of 6:55 bypasses sleepy pattern-recognition circuitry for roughly two weeks.`,
      },
      {
        cues: ['airport', 'boarding', 'group', 'line', 'queue'],
        layout: 'three_panel',
        prop: 'whiteboard',
        cast: ['cueball', 'megan'],
        poses: [['pointing', 'conversing'], ['conversing', 'arms_crossed'], ['shrug', 'pointing']],
        turns: [
          { speaker: 'left', text: 'NOW BOARDING DIAMOND ULTRA GROUP ZERO-MINUS-ALPHA!' },
          { speaker: 'right', text: 'EVERYONE STILL LANDS AT THE EXACT SAME MINUTE, RIGHT?' },
          { speaker: 'left', text: 'YES, BUT STANDING IN A CROWDED JETBRIDGE FEELS EXCLUSIVE.' },
        ],
        alt: `Back-to-front window-middle-aisle beats status boarding by 40%, which is why airlines treat it as heresy.`,
      },
      {
        cues: ['painting', 'ceiling', 'speckles', 'glasses', 'drip'],
        layout: 'two_panel',
        prop: 'pedestal',
        cast: ['cueball', 'ponytail'],
        poses: [['pointing', 'conversing'], ['hands_on_head', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: `I TAPED EVERY TRIM EDGE FOR THREE HOURS BEFORE PAINTING ${np}.` },
          { speaker: 'right', text: 'ZERO DROPS ON THE FLOOR AND FORTY-TWO WHITE MICRO-DOTS ON YOUR GLASSES.' },
        ],
        alt: `Paint rollers obey centrifugal dispersion specifically calibrated to curve around drop cloths.`,
      },
      {
        cues: ['water', 'bottle', 'artisanal', 'spring', 'station', 'overpriced'],
        layout: 'two_panel',
        prop: 'pedestal',
        cast: ['megan', 'black_hat'],
        poses: [['pointing', 'arms_crossed'], ['conversing', 'shrug']],
        turns: [
          { speaker: 'left', text: 'NINE DOLLARS FOR GLACIAL ARTISANAL WATER AT THE STATION KIOSK?!' },
          { speaker: 'right', text: 'WE BOTTLE MUNICIPAL TAP WATER AND MONETIZE CAPTIVE THIRST.' },
        ],
        alt: `'Filtered through ancient volcanic rock' is technically true of all water on Earth since the Hadean eon.`,
      },
      {
        cues: ['repair', 'leaky', 'faucet', 'tape', 'plumber', 'plumbing'],
        layout: 'three_panel',
        prop: 'pedestal',
        cast: ['cueball', 'megan'],
        poses: [['pointing', 'conversing'], ['hands_on_head', 'pointing'], ['shrug', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: `TEN-MINUTE DIY FIX FOR ${np}: ONE ROLL OF TEFLON TAPE.` },
          { speaker: 'right', text: 'YOU HAVE MADE THREE HARDWARE STORE TRIPS AND SHUT OFF THE STREET MAIN.' },
          { speaker: 'left', text: 'TRIP FOUR IS PURE VICTORY LAP, I CAN FEEL IT.' },
        ],
        alt: `No home plumbing repair officially begins until you own two incompatible brass adapters and a damp towel.`,
      },
    ];
  }

  if (domain === 'net_hw') {
    return [
      {
        cues: ['printer', 'smells', 'fear', 'deadline', 'ink', 'cyan'],
        layout: 'three_panel',
        prop: 'server_rack',
        cast: ['cueball', 'black_hat'],
        poses: [['pointing', 'conversing'], ['hands_on_head', 'arms_crossed'], ['shrug', 'pointing']],
        turns: [
          { speaker: 'left', text: 'I JUST NEED TO PRINT ONE BLACK-AND-WHITE PAGE BEFORE MY DEADLINE!' },
          { speaker: 'right', text: 'ERROR: CANNOT PRINT BLACK TEXT BECAUSE MAGENTA IS FEELING MELANCHOLY.' },
          { speaker: 'left', text: 'IT HAS A CORTISOL SENSOR AIMED AT HUMANS, DOESNT IT?' },
        ],
        alt: `Printers were sent back in time to prevent paperless offices AND peaceful mornings simultaneously.`,
      },
      {
        cues: ['domain', '.dev', '1am', 'buying'],
        layout: 'two_panel',
        prop: 'laptop_desk',
        cast: ['cueball', 'ponytail'],
        poses: [['typing', 'conversing'], ['shrug', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: '1:12 AM IDEA ACQUIRED! I JUST BOUGHT THE PERFECT .DEV DOMAIN FOR $14!' },
          { speaker: 'right', text: 'WELCOME TO THE GRAVEYARD OF FORTY-TWO UNBUILT SIDE PROJECTS.' },
        ],
        alt: `Buying the domain releases 95% of the dopamine of shipping the app with 0% of the bug reports.`,
      },
      {
        cues: ['wikipedia', 'pottery', 'submarines', 'click', 'chain'],
        layout: 'three_panel',
        prop: 'laptop_desk',
        cast: ['megan', 'cueball'],
        poses: [['typing', 'conversing'], ['pointing', 'chin_think'], ['hands_on_head', 'shrug']],
        turns: [
          { speaker: 'left', text: 'I OPENED WIKIPEDIA TO CHECK WHEN BRONZE POTTERY STARTED.' },
          { speaker: 'right', text: 'FOUR MINUTES LATER?' },
          { speaker: 'left', text: 'READING DECLASSIFIED SOVIET TITANIUM SUBMARINE HULL METALLURGY.' },
        ],
        alt: `Every Wikipedia article is at most five blue links away from either Plate Tectonics or Charlemagne.`,
      },
      {
        cues: ['smart tv', 'tv', '18', 'hdmi', 'linux'],
        layout: 'two_panel',
        prop: 'couch',
        cast: ['cueball', 'megan'],
        poses: [['pointing', 'conversing'], ['hands_on_head', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: 'A 1995 TV SWITCHED INPUTS AT THE SPEED OF AN ELECTRICAL RELAY.' },
          { speaker: 'right', text: 'OURS TAKES EIGHTEEN SECONDS TO BOOT AN OS AND SHOW US AN AD FOR ITSELF.' },
        ],
        alt: `Never connect a television to Wi-Fi unless you want HDMI 2 gated behind a terms-of-service modal.`,
      },
      {
        cues: ['forum', '2009', 'fixed', 'never', 'mind', 'thread'],
        layout: 'two_panel',
        prop: 'laptop_desk',
        cast: ['ponytail', 'cueball'],
        poses: [['typing', 'pointing'], ['hands_on_head', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: 'ONE PERSON IN 2009 HAD MY EXACT OBSCURE ERROR MESSAGE!' },
          { speaker: 'right', text: 'ONLY REPLY, SAME USER: "NVM FIXED IT :)" — THREAD LOCKED.' },
        ],
        alt: `WHO WERE YOU, DENVERCODER9? WHAT DID YOU SEE?!`,
      },
      {
        cues: ['usb-a', 'flips', 'four-dimensional', 'plug', 'usb'],
        layout: 'two_panel',
        prop: 'whiteboard',
        cast: ['cueball', 'megan'],
        poses: [['pointing', 'conversing'], ['shrug', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: 'TRY 1: WRONG. FLIP 180 DEGREES: STILL WRONG. FLIP AGAIN: GOES IN.' },
          { speaker: 'right', text: 'USB-A PLUGS HAVE SPIN-1/2 QUANTUM SYMMETRY.' },
        ],
        alt: `You have to rotate a spinor by 720 degrees to return to the original state, which explains every back-of-PC cable reach.`,
      },
      {
        cues: ['unsubscribe', 'log', 'preferences', 'manage'],
        layout: 'two_panel',
        prop: 'laptop_desk',
        cast: ['megan', 'black_hat'],
        poses: [['typing', 'conversing'], ['pointing', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: 'TO UNSUBSCRIBE FROM OUR EMAILS, PLEASE LOG IN OR RESET YOUR PASSWORD.' },
          { speaker: 'right', text: 'CONGRATS, THAT BUTTON JUST TRAINED MY SPAM FILTER FOR EVERYONE.' },
        ],
        alt: `'Allow up to 10 business days to stop emailing you' implies a clerk manually removing index cards with tweezers.`,
      },
      {
        cues: ['captive', 'portal', 'hotel', 'localhost'],
        layout: 'two_panel',
        prop: 'laptop_desk',
        cast: ['ponytail', 'cueball'],
        poses: [['typing', 'conversing'], ['hands_on_head', 'arms_crossed']],
        turns: [
          { speaker: 'left', text: 'HOTEL WI-FI INTERCEPTED MY HTTPS REQUEST AND HIJACKED LOCALHOST!' },
          { speaker: 'right', text: 'ALWAYS TYPE `NEVERSLAIN.COM` LIKE A PILGRIM SUMMONING PORT 80.' },
        ],
        alt: `Captive portals operate in a lawless pocket universe between OSI Layer 3 and hospitality billing.`,
      },
      {
        cues: ['autocorrect', 'ducking', 'spelled'],
        layout: 'two_panel',
        prop: 'laptop_desk',
        cast: ['cueball', 'ponytail'],
        poses: [['typing', 'conversing'], ['hands_on_head', 'pointing']],
        turns: [
          { speaker: 'left', text: `TEN BILLION PARAMETERS ON MY PHONE AND ${np} STILL CHANGED IT TO "DUCKING".` },
          { speaker: 'right', text: 'STATISTICALLY, ZERO ANGRY HUMANS HAVE EVER TEXTED ABOUT WATERFOWL.' },
        ],
        alt: `After nine years of correcting it back, my keyboard finally compromised by misspelling my own name instead.`,
      },
    ];
  }

  // Open-World Compositional Randall-Munroe Blueprints (pure general-purpose, zero narrow cue hijacks!)
  return [
    {
      cues: [],
      layout: 'three_panel',
      prop: 'whiteboard',
      cast: ['megan', 'cueball'],
      poses: [['pointing', 'chin_think'], ['conversing', 'pointing'], ['arms_crossed', 'shrug']],
      turns: [
        { speaker: 'left', text: `I BUILT A MATHEMATICAL MODEL OF ${np} FROM FIRST PRINCIPLES.` },
        { speaker: 'right', text: `DOES IT PREDICT ${ctx || 'REAL-WORLD BEHAVIOR'}?` },
        { speaker: 'left', text: 'PERFECTLY, AS LONG AS HUMANS ACT RATIONALLY AND FRICTION IS ZERO.' },
      ],
      alt: `Adding one empirical measurement ruined the R-squared of ${npLow}, so we declared reality an outlier.`,
    },
    {
      cues: [],
      layout: 'two_panel',
      prop: 'laptop_desk',
      cast: ['cueball', 'ponytail'],
      poses: [['typing', 'conversing'], ['hands_on_head', 'arms_crossed']],
      turns: [
        { speaker: 'left', text: `HOW COMPLICATED COULD ${np} POSSIBLY BE?` },
        { speaker: 'right', text: `FAMOUS LAST WORDS BEFORE DISCOVERING ${head} HAS THREE COMPETING ISO STANDARDS.` },
      ],
      alt: `Every seemingly simple topic has a 400-page forum where enthusiasts argue bitterly about ${npLow} tolerances.`,
    },
    {
      cues: [],
      layout: 'three_panel',
      prop: 'pointer_stick',
      cast: ['hairbun', 'beret_guy'],
      poses: [['pointer_rod', 'conversing'], ['conversing', 'chin_think'], ['arms_crossed', 'shrug']],
      turns: [
        { speaker: 'left', text: `IF WE EXTRAPOLATE ${np} LINEARLY, BY THURSDAY IT EXCEEDS THE MASS OF JUPITER.` },
        { speaker: 'right', text: 'SHOULD WE USE A LOG SCALE?' },
        { speaker: 'left', text: 'NO, LINEAR CHARTS GET WAY MORE FUNDING.' },
      ],
      alt: `Fitting a seventh-degree polynomial to two data points on ${npLow} predicts brief time travel next month.`,
    },
    {
      cues: [],
      layout: 'single_panel',
      prop: 'couch',
      cast: ['cueball', 'black_hat'],
      turns: [
        { speaker: 'left', text: `WHY DOES NOBODY FIX ${np} PROPERLY?` },
        { speaker: 'right', text: 'BECAUSE THE TEMPORARY WORKAROUND FROM 2014 NOW HAS A FAN CLUB.' },
        { speaker: 'left', text: 'AND TWO BOOKS EXPLAINING ITS QUIRKS AS BEST PRACTICES.' },
      ],
      alt: `Give any makeshift hack ten years and someone will defend ${npLow} as timeless heritage craftsmanship.`,
    },
    {
      cues: [],
      layout: 'four_panel',
      prop: 'laptop_desk',
      cast: ['cueball', 'megan'],
      poses: [['typing', 'conversing'], ['pointing', 'chin_think'], ['hands_on_head', 'pointing'], ['shrug', 'arms_crossed']],
      turns: [
        { speaker: 'left', text: `I BUILT A SPREADSHEET TO OPTIMIZE ${np}.` },
        { speaker: 'right', text: 'DID IT SAVE ANY TIME?' },
        { speaker: 'left', text: 'NO, MAINTAINING THE PIVOT TABLES TAKES FOUR HOURS A WEEK.' },
        { speaker: 'right', text: 'WORTH IT?', reply: 'THE CONDITIONAL FORMATTING IS GORGEOUS.' },
      ],
      alt: `Any personal problem can be transformed into an enterprise data-entry problem with enough columns.`,
    },
    {
      cues: [],
      layout: 'two_panel',
      prop: 'pedestal',
      cast: ['beret_guy', 'ponytail'],
      poses: [['pointing', 'conversing'], ['shrug', 'arms_crossed']],
      turns: [
        { speaker: 'left', text: `I RELEASED ${np} INTO THE WILD SO ${up(sem.itThey)} COULD ROAM FREE.` },
        { speaker: 'right', text: `THAT EXPLAINS WHY ${ctx || 'THE PARKING LOT'} IS CURRENTLY QUARANTINED.` },
      ],
      alt: `Nature is healing, except ${npLow} quickly learned how to open latching trash cans.`,
    },
    {
      cues: [],
      layout: 'three_panel',
      prop: 'whiteboard',
      cast: ['ponytail', 'cueball'],
      poses: [['pointing', 'conversing'], ['conversing', 'chin_think'], ['arms_crossed', 'hands_on_head']],
      turns: [
        { speaker: 'left', text: `THERMODYNAMICALLY, ${np} VIOLATES AT LEAST TWO CONSERVATION LAWS.` },
        { speaker: 'right', text: 'YET IT HAPPENS EVERY SINGLE TUESDAY.' },
        { speaker: 'left', text: 'MAXWELLS DEMON WAS ACTUALLY JUST TRYING TO MANAGE THIS.' },
      ],
      alt: `Local entropy always increases fastest right when you are carrying two cups of hot coffee.`,
    },
    {
      cues: [],
      layout: 'two_panel',
      prop: 'couch',
      cast: ['megan', 'cueball'],
      poses: [['conversing', 'pointing'], ['arms_crossed', 'shrug']],
      turns: [
        { speaker: 'left', text: `HUMANS EVOLVED TO TRACK GAZELLES, NOT DEAL WITH ${np}.` },
        { speaker: 'right', text: 'MY AMYGDALA TREATS BOTH AS AN APEX PREDATOR IN THE BUSHES.' },
      ],
      alt: `Fight-or-flight hormones are noticeably less helpful when the threat is ${npLow} ${ctx ? `during ${low(ctx)}` : 'at 9 AM'}.`,
    },
    {
      cues: [],
      layout: 'three_panel',
      prop: 'laptop_desk',
      cast: ['cueball', 'black_hat'],
      poses: [['typing', 'conversing'], ['conversing', 'arms_crossed'], ['hands_on_head', 'pointing']],
      turns: [
        { speaker: 'left', text: `IS THERE ANY LEGAL RULE AGAINST ${np}?` },
        { speaker: 'right', text: 'NOT YET, WHICH MEANS WE HAVE UNTIL FRIDAY TO SET THE RECORD.' },
        { speaker: 'left', text: 'PLEASE DO NOT GET ME MENTIONED IN A SAFETY BRIEFING AGAIN.' },
      ],
      alt: `Every oddly specific warning sign in a hotel lobby commemorates one person's creative approach to ${npLow}.`,
    },
    {
      cues: [],
      layout: 'two_panel',
      prop: 'server_rack',
      cast: ['hairbun', 'cueball'],
      poses: [['pointing', 'conversing'], ['arms_crossed', 'shrug']],
      turns: [
        { speaker: 'left', text: `WE FORMED A SUBCOMMITTEE TO STANDARDIZE ${np}.` },
        { speaker: 'right', text: 'SIX MONTHS LATER WE ONLY AGREED ON THE FONT FOR THE COVER PAGE.' },
      ],
      alt: `Appendix B still contains an active schism over whether ${npLow} should be zero-indexed.`,
    },
    {
      cues: [],
      layout: 'three_panel',
      prop: 'pedestal',
      cast: ['cueball', 'megan'],
      poses: [['pointing', 'conversing'], ['chin_think', 'pointing'], ['shrug', 'arms_crossed']],
      turns: [
        { speaker: 'left', text: `I BOUGHT SPECIALIZED PRO-GRADE GEAR FOR ${np}.` },
        { speaker: 'right', text: 'HOW OFTEN DO YOU ACTUALLY DO THAT?' },
        { speaker: 'left', text: 'TWICE A DECADE, BUT THOSE TEN MINUTES WILL BE AEROSPACE GRADE.' },
      ],
      alt: `Amortized over two uses, each session of ${npLow} cost roughly $190 per minute.`,
    },
    {
      cues: [],
      layout: 'two_panel',
      prop: 'whiteboard',
      cast: ['cueball', 'ponytail'],
      poses: [['pointing', 'conversing'], ['hands_on_head', 'pointing']],
      turns: [
        { speaker: 'left', text: `AT SMALL SCALE ${np} SEEMS TOTALLY HARMLESS.` },
        { speaker: 'right', text: 'MULTIPLY BY EIGHT BILLION PEOPLE AND IT ACCOUNTS FOR 2% OF GLOBAL ELECTRICITY.' },
      ],
      alt: `Fermi estimation turns every casual daydream about ${npLow} into an urgent civil-engineering problem.`,
    },
    {
      cues: [],
      layout: 'single_panel',
      prop: 'laptop_desk',
      cast: ['megan', 'ponytail'],
      turns: [
        { speaker: 'left', text: `NOBODY ACTUALLY ENJOYS ${np}, RIGHT?` },
        { speaker: 'right', text: 'NO, EVERYONE JUST PRETENDS TO BECAUSE THEY THINK EVERYONE ELSE DOES.' },
        { speaker: 'left', text: 'ABELINE PARADOX STRIKES AGAIN.' },
      ],
      alt: `Half of polite society is eight people silently enduring ${npLow} so nobody seems difficult.`,
    },
    {
      cues: [],
      layout: 'four_panel',
      prop: 'whiteboard',
      cast: ['cueball', 'megan'],
      poses: [['pointing', 'conversing'], ['conversing', 'chin_think'], ['pointing', 'arms_crossed'], ['hands_on_head', 'shrug']],
      turns: [
        { speaker: 'left', text: `STAGE 1 OF ${head}: "THIS WILL ONLY TAKE FIVE MINUTES."` },
        { speaker: 'left', text: 'STAGE 2: DISCOVERING THREE MISSING TOOLS AND AN OBSCURE TERMINOLOGY.' },
        { speaker: 'right', text: 'WHAT IS STAGE 3?' },
        { speaker: 'left', text: 'WATCHING A 2012 YOUTUBE TUTORIAL AT 2X SPEED IN SILENT DESPAIR.' },
      ],
      alt: `The video always has a 45-second dubstep intro and the exact solution filmed slightly out of frame.`,
    },
    {
      cues: [],
      layout: 'two_panel',
      prop: 'pointer_stick',
      cast: ['ponytail', 'beret_guy'],
      poses: [['pointer_rod', 'conversing'], ['arms_crossed', 'shrug']],
      turns: [
        { speaker: 'left', text: `STATISTICALLY, ${np} CORRELATES 0.94 WITH ICE CREAM SALES.` },
        { speaker: 'right', text: 'SO BANNING VANILLA CONES WILL SOLVE IT ONCE AND FOR ALL!' },
      ],
      alt: `Confounding variables are just spicy seasoning for people who already made their slide deck.`,
    },
    {
      cues: [],
      layout: 'three_panel',
      prop: 'couch',
      cast: ['cueball', 'megan'],
      poses: [['conversing', 'pointing'], ['hands_on_head', 'conversing'], ['shrug', 'arms_crossed']],
      turns: [
        { speaker: 'left', text: `I PROMISED MYSELF I WOULD DEAL WITH ${np} LIKE A CALM ADULT.` },
        { speaker: 'right', text: 'HOW LONG DID THAT LAST?' },
        { speaker: 'left', text: 'FOUR SECONDS, THEN I BARGAINED WITH INANIMATE MATTER OUT LOUD.' },
      ],
      alt: `Apologizing politely to ${npLow} still works slightly more often than yelling at ${sem.themIt}.`,
    },
    {
      cues: [],
      layout: 'two_panel',
      prop: 'laptop_desk',
      cast: ['black_hat', 'cueball'],
      poses: [['typing', 'conversing'], ['arms_crossed', 'hands_on_head']],
      turns: [
        { speaker: 'left', text: `IF YOU REFRAME ${np} AS PERFORMANCE ART, ZERO MISTAKES WERE MADE.` },
        { speaker: 'right', text: 'THE FIRE MARSHAL DID NOT ACCEPT THAT GENRE CLASSIFICATION.' },
      ],
      alt: `Adding a small museum placard in Helvetica next to ${npLow} buys you at least twenty minutes of respect.`,
    },
    {
      cues: [],
      layout: 'three_panel',
      prop: 'server_rack',
      cast: ['ponytail', 'cueball'],
      poses: [['pointing', 'conversing'], ['chin_think', 'pointing'], ['arms_crossed', 'shrug']],
      turns: [
        { speaker: 'left', text: `EVERY GENERATION REINVENTS ${np} AND CALLS IT A REVOLUTION.` },
        { speaker: 'right', text: 'DID ANCIENT ROME HAVE THIS PROBLEM TOO?' },
        { speaker: 'left', text: 'PLINY THE ELDER COMPLAINED ABOUT IT IN BOOK SEVEN.' },
      ],
      alt: `Translating 1st-century Latin graffiti reveals surprisingly modern opinions about ${npLow}.`,
    },
    {
      cues: [],
      layout: 'two_panel',
      prop: 'pedestal',
      cast: ['megan', 'cueball'],
      poses: [['pointing', 'conversing'], ['shrug', 'arms_crossed']],
      turns: [
        { speaker: 'left', text: `THE OFFICIAL INSTRUCTIONS FOR ${np} HAVE ONLY THREE STEPS.` },
        { speaker: 'right', text: 'STEP 2 IS DOING ALL THE IMPOSSIBLE PARTS WITHOUT DROPPING ANYTHING.' },
      ],
      alt: `'Simply align the tabs and click into place' is the biggest lie in technical prose.`,
    },
    {
      cues: [],
      layout: 'single_panel',
      prop: 'whiteboard',
      cast: ['cueball', 'ponytail', 'black_hat'],
      turns: [
        { speaker: 'left', text: `WE SURVEYED 500 PEOPLE ABOUT ${np}.` },
        { speaker: 'right', text: 'WHAT WAS THE TOP FINDING?' },
        { speaker: 'third', text: '100% OF RESPONDENTS HATE ANSWERING SURVEYS DURING DINNER.' },
      ],
      alt: `Selection bias means all consumer data really just measures people who answer unknown phone numbers.`,
    },
  ];
}

export function synthesizeHumorSpec(promptText, iconicTemplates = []) {
  const clean = String(promptText || '').trim();
  const lowText = clean.toLowerCase();

  for (const orig of ORIGINAL_SYNTHESIS_BANK) {
    if (orig.match.some(k => lowText.includes(k))) {
      return orig;
    }
  }
  for (const tpl of iconicTemplates) {
    if (tpl.match.some(k => lowText.includes(k))) {
      return tpl;
    }
  }

  const sem = parsePromptSemantics(clean);
  const domain = classifyPromptDomain(sem);
  const candidates = buildDomainBlueprints(domain, sem) || buildRemainingBlueprints(domain, sem);
  const chosen = pickBlueprint(candidates, sem);

  return {
    title: sem.title,
    scene: 'kinematic',
    layout: chosen.layout,
    prop: chosen.prop,
    cast: chosen.cast,
    poses: chosen.poses || null,
    chart_meta: chosen.chart_meta || null,
    turns: chosen.turns,
    alt: chosen.alt,
  };
}
