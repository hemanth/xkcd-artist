#!/usr/bin/env node
/**
 * XkcdHumorEval: Quantitative Dialogue Quality, Diversity, Semantic Grounding & Wit Benchmark
 * Evaluates comic synthesis across N = 120 diverse & adversarial unseen prompts with B = 1,000 bootstrap 95% CIs.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import xkcdArtist from '../js/src/index.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// 120-prompt Golden Benchmark across 15 slices (8 prompts per slice), including substring-collision traps!
export const GOLDEN_PROMPTS = [
  // 1. Substring-collision traps (words containing 'pr', 'ai', 'git', 'type', 'css', 'cloud' inside unrelated words!)
  { slice: 'trap_substrings', expectedDomain: 'non_code', prompt: 'Why the office printer smells fear right before a deadline' },
  { slice: 'trap_substrings', expectedDomain: 'non_code', prompt: 'Dialing in espresso grind size before 7am' },
  { slice: 'trap_substrings', expectedDomain: 'non_code', prompt: 'Waiting in an airport boarding group line that makes no geometric sense' },
  { slice: 'trap_substrings', expectedDomain: 'non_code', prompt: 'Painting a bedroom ceiling without getting speckles on my glasses' },
  { slice: 'trap_substrings', expectedDomain: 'non_code', prompt: 'Explaining compound interest to a golden retriever' },
  { slice: 'trap_substrings', expectedDomain: 'non_code', prompt: 'Buying an overpriced artisanal spring water bottle at the train station' },
  { slice: 'trap_substrings', expectedDomain: 'non_code', prompt: 'Trying to repair a leaky kitchen faucet with plumber tape' },
  { slice: 'trap_substrings', expectedDomain: 'non_code', prompt: 'Why cumulus clouds look flat on the bottom' },

  // 2. AI / LLMs / Agents (8 distinct sub-scenarios that MUST NOT all generate the same comic!)
  { slice: 'ai_llm', expectedDomain: 'ai', prompt: 'Training neural agents to fold fitted bedsheets' },
  { slice: 'ai_llm', expectedDomain: 'ai', prompt: 'Writing a 600-word prompt so an LLM summarizes a three-line email' },
  { slice: 'ai_llm', expectedDomain: 'ai', prompt: 'Two customer support chatbots politely apologizing to each other forever' },
  { slice: 'ai_llm', expectedDomain: 'ai', prompt: 'When an autocomplete model invents a fake Python library and you pip install it anyway' },
  { slice: 'ai_llm', expectedDomain: 'ai', prompt: 'Fine-tuning a local language model on my own Slack messages' },
  { slice: 'ai_llm', expectedDomain: 'ai', prompt: 'AI benchmark leaderboard where every model scores 99.9% except on basic arithmetic' },
  { slice: 'ai_llm', expectedDomain: 'ai', prompt: 'Asking a coding assistant to fix one bug and getting a complete rewrite' },
  { slice: 'ai_llm', expectedDomain: 'ai', prompt: 'Voice assistant mishearing my kitchen timer as an existential question' },

  // 3. Infra / Cloud / Containers
  { slice: 'infra_cloud', expectedDomain: 'infra', prompt: 'Deploying a static personal blog on a 14-node Kubernetes cluster' },
  { slice: 'infra_cloud', expectedDomain: 'infra', prompt: 'Getting a surprise $4,800 cloud bill from an unattached NAT gateway' },
  { slice: 'infra_cloud', expectedDomain: 'infra', prompt: 'Debugging YAML indentation at 2am when spaces vs tabs break prod' },
  { slice: 'infra_cloud', expectedDomain: 'infra', prompt: 'Serverless cold start taking longer than driving to the datacenter' },
  { slice: 'infra_cloud', expectedDomain: 'infra', prompt: 'Migrating three microservices back into one SQLite file' },
  { slice: 'infra_cloud', expectedDomain: 'infra', prompt: 'Docker build failing because an Alpine mirror from 2018 vanished' },
  { slice: 'infra_cloud', expectedDomain: 'infra', prompt: 'Terraform plan wanting to destroy 47 resources for a one-character tag typo' },
  { slice: 'infra_cloud', expectedDomain: 'infra', prompt: 'Self-hosting a home NAS that uses more electricity than the fridge' },

  // 4. Code / Compilers / CSS / Regex
  { slice: 'code_craft', expectedDomain: 'code', prompt: 'Centering a div in CSS breaks the footer on mobile Safari' },
  { slice: 'code_craft', expectedDomain: 'code', prompt: 'Writing a regular expression that worked yesterday and looks like hieroglyphs today' },
  { slice: 'code_craft', expectedDomain: 'code', prompt: 'Spending forty minutes naming a boolean variable' },
  { slice: 'code_craft', expectedDomain: 'code', prompt: 'Rust borrow checker rejecting my linked list for the ninth time' },
  { slice: 'code_craft', expectedDomain: 'code', prompt: 'Deleting dead code from 2016 and breaking the billing pipeline' },
  { slice: 'code_craft', expectedDomain: 'code', prompt: 'TypeScript generic error message scrolling for six pages' },
  { slice: 'code_craft', expectedDomain: 'code', prompt: 'Adding print statements because the debugger itself crashed' },
  { slice: 'code_craft', expectedDomain: 'code', prompt: 'Code comment saying do not remove this sleep(100) under any circumstances' },

  // 5. Git & Incidents & DNS
  { slice: 'git_incidents', expectedDomain: 'ops', prompt: 'Pushing a one-line config tweak at 4:55pm on Friday' },
  { slice: 'git_incidents', expectedDomain: 'ops', prompt: 'Resolving a git rebase conflict across 90 files by starting a fresh clone' },
  { slice: 'git_incidents', expectedDomain: 'ops', prompt: 'SSL certificate expiring on a leap year Sunday at midnight' },
  { slice: 'git_incidents', expectedDomain: 'ops', prompt: 'Spending six hours debugging the network before realizing it was DNS' },
  { slice: 'git_incidents', expectedDomain: 'ops', prompt: 'Git commit history going from clean messages to fix, fix2, asdfgh' },
  { slice: 'git_incidents', expectedDomain: 'ops', prompt: 'Status page showing all green icons while production is on fire' },
  { slice: 'git_incidents', expectedDomain: 'ops', prompt: 'PagerDuty alert waking me up for a test environment health check' },
  { slice: 'git_incidents', expectedDomain: 'ops', prompt: 'Force pushing to main right before going into Airplane Mode' },

  // 6. Annotated Charts
  { slice: 'charts', expectedDomain: 'chart', prompt: 'Chart of developer confidence vs years spent working with timezones' },
  { slice: 'charts', expectedDomain: 'chart', prompt: 'Graph of open browser tabs vs probability I ever read them' },
  { slice: 'charts', expectedDomain: 'chart', prompt: 'Plot of recipe blog word count vs distance to actual ingredients' },
  { slice: 'charts', expectedDomain: 'chart', prompt: 'Distribution curve of microwave popcorn between unpopped kernels and charcoal' },
  { slice: 'charts', expectedDomain: 'chart', prompt: 'Chart of meeting headcount vs decisions actually made' },
  { slice: 'charts', expectedDomain: 'chart', prompt: 'Graph of battery percentage anxiety from 100% down to the immortal last 1%' },
  { slice: 'charts', expectedDomain: 'chart', prompt: 'Correlation plot between monitor count and time spent rearranging windows' },
  { slice: 'charts', expectedDomain: 'chart', prompt: 'Curve of side project enthusiasm vs domain name renewals' },

  // 7. Flowcharts
  { slice: 'flowcharts', expectedDomain: 'flowchart', prompt: 'Flowchart for deciding whether to automate a 30-second task' },
  { slice: 'flowcharts', expectedDomain: 'flowchart', prompt: 'Decision tree for troubleshooting a mysterious noise in my car' },
  { slice: 'flowcharts', expectedDomain: 'flowchart', prompt: 'Flowchart for deciding if this 45-minute meeting could have been an email' },
  { slice: 'flowcharts', expectedDomain: 'flowchart', prompt: 'How to decide cheat sheet for replying to someone wrong on a forum' },
  { slice: 'flowcharts', expectedDomain: 'flowchart', prompt: 'Workflow diagram for fixing family Wi-Fi over Thanksgiving dinner' },
  { slice: 'flowcharts', expectedDomain: 'flowchart', prompt: 'Flowchart for choosing between sleeping early and reading Wikipedia at 1am' },
  { slice: 'flowcharts', expectedDomain: 'flowchart', prompt: 'Decision tree for throwing away an unlabeled power brick from 2011' },
  { slice: 'flowcharts', expectedDomain: 'flowchart', prompt: 'Flowchart for debugging why Bluetooth headphones connected to the wrong room' },

  // 8. X vs Y Comparisons
  { slice: 'comparisons', expectedDomain: 'vs', prompt: 'Cold brew coffee vs triple espresso during a Monday standup' },
  { slice: 'comparisons', expectedDomain: 'vs', prompt: 'Reading documentation for 5 minutes vs debugging blind for 6 hours' },
  { slice: 'comparisons', expectedDomain: 'vs', prompt: 'Mechanical keyboard clicks vs open office acoustic sanity' },
  { slice: 'comparisons', expectedDomain: 'vs', prompt: 'Packing a suitcase 3 days early vs shoving clothes in 12 minutes before the cab' },
  { slice: 'comparisons', expectedDomain: 'vs', prompt: 'SQLite on one NVMe drive vs distributed NoSQL across three continents' },
  { slice: 'comparisons', expectedDomain: 'vs', prompt: 'Buying books vs actually finishing the stack on my nightstand' },
  { slice: 'comparisons', expectedDomain: 'vs', prompt: 'Smart home light switches vs regular wall switches when guests visit' },
  { slice: 'comparisons', expectedDomain: 'vs', prompt: 'Estimating a task takes two hours vs discovering undocumented legacy APIs' },

  // 9. Physics / Space / Math
  { slice: 'science_math', expectedDomain: 'science', prompt: 'Calculating orbital mechanics using kerbal intuition and a napkin' },
  { slice: 'science_math', expectedDomain: 'science', prompt: 'Converting imperial recipes between cups, ounces, and relativistic mass' },
  { slice: 'science_math', expectedDomain: 'science', prompt: 'Thermodynamics of blowing on hot soup vs stirring it' },
  { slice: 'science_math', expectedDomain: 'science', prompt: 'Proof by intimidation in an undergraduate linear algebra lecture' },
  { slice: 'science_math', expectedDomain: 'science', prompt: 'Trying to synchronize clocks across Earth, the Moon, and Daylight Saving Time' },
  { slice: 'science_math', expectedDomain: 'science', prompt: 'Bayesian probability that my lost keys are in the fridge' },
  { slice: 'science_math', expectedDomain: 'science', prompt: 'Physicist approximating a horse as a frictionless cylinder' },
  { slice: 'science_math', expectedDomain: 'science', prompt: 'Astronomers naming a terrifying exoplanet something like HD-40291b' },

  // 10. Security / Passwords / Privacy
  { slice: 'security', expectedDomain: 'security', prompt: 'Password policy requiring a hieroglyph, a prime number, and your childhood fear' },
  { slice: 'security', expectedDomain: 'security', prompt: 'Two-factor authentication code sitting on my phone across the room' },
  { slice: 'security', expectedDomain: 'security', prompt: 'Cookie consent banner with 340 ad partners under legitimate interest' },
  { slice: 'security', expectedDomain: 'security', prompt: 'Rotating SSH keys and immediately locking myself out of the jump box' },
  { slice: 'security', expectedDomain: 'security', prompt: 'Phishing training email that looks more legitimate than real HR emails' },
  { slice: 'security', expectedDomain: 'security', prompt: 'End-to-end encrypting a grocery list containing only milk and limes' },
  { slice: 'security', expectedDomain: 'security', prompt: 'Security audit finding the admin password taped to a monitor' },
  { slice: 'security', expectedDomain: 'security', prompt: 'Smartfridge joining a botnet because nobody patched the ice maker' },

  // 11. Work / Meetings / Corporate Life
  { slice: 'work_meetings', expectedDomain: 'work', prompt: 'Pre-meeting sync to align on the agenda for Friday alignment sync' },
  { slice: 'work_meetings', expectedDomain: 'work', prompt: 'Jira ticket marked low priority since the Obama administration' },
  { slice: 'work_meetings', expectedDomain: 'work', prompt: 'Saying let us take this offline and never speaking of it again' },
  { slice: 'work_meetings', expectedDomain: 'work', prompt: 'Sharing my screen and accidentally revealing 190 chaotic browser tabs' },
  { slice: 'work_meetings', expectedDomain: 'work', prompt: 'Spreadsheet from finance that runs the entire company on nested VLOOKUPs' },
  { slice: 'work_meetings', expectedDomain: 'work', prompt: 'Calendar with one 15-minute gap all day labeled focus time' },
  { slice: 'work_meetings', expectedDomain: 'work', prompt: 'Unmuting on a 60-person video call just to say sorry go ahead' },
  { slice: 'work_meetings', expectedDomain: 'work', prompt: 'Quarterly roadmap slide where Q4 is just a box labeled magic' },

  // 12. Food / Coffee / Cooking
  { slice: 'food_coffee', expectedDomain: 'food', prompt: 'Avocado ripeness window lasting approximately 11 minutes on Tuesday' },
  { slice: 'food_coffee', expectedDomain: 'food', prompt: 'Buying kale with ambitious Sunday meal-prep optimism' },
  { slice: 'food_coffee', expectedDomain: 'food', prompt: 'Sourdough starter demanding more daily care than a houseplant' },
  { slice: 'food_coffee', expectedDomain: 'food', prompt: 'Boiling pasta and still checking the box for 9 to 11 minutes every single time' },
  { slice: 'food_coffee', expectedDomain: 'food', prompt: 'Ordering spicy Thai food at level 4 and regretting hubris' },
  { slice: 'food_coffee', expectedDomain: 'food', prompt: 'Hotel breakfast waffle iron engineering tolerances' },
  { slice: 'food_coffee', expectedDomain: 'food', prompt: 'Drinking decaf at 9pm and still staring at the ceiling until 2am' },
  { slice: 'food_coffee', expectedDomain: 'food', prompt: 'Kitchen drawer jammed shut by one sideways potato masher' },

  // 13. Cats / Pets / Nature
  { slice: 'pets_nature', expectedDomain: 'pets', prompt: 'Cat ignoring a $90 velvet bed to sit inside an Amazon cardboard box' },
  { slice: 'pets_nature', expectedDomain: 'pets', prompt: 'Houseplant dying from 2 milliliters too much water while sidewalk weeds thrive in concrete' },
  { slice: 'pets_nature', expectedDomain: 'pets', prompt: 'Crow solving a multi-step puzzle box faster than my morning brain' },
  { slice: 'pets_nature', expectedDomain: 'pets', prompt: 'Cat sprinting across the hallway at 3:14am for conservation of zoomies' },
  { slice: 'pets_nature', expectedDomain: 'pets', prompt: 'Dog treating the vacuum cleaner like an eldritch invader' },
  { slice: 'pets_nature', expectedDomain: 'pets', prompt: 'Seagull calculating projectile trajectories at a beach picnic' },
  { slice: 'pets_nature', expectedDomain: 'pets', prompt: 'Mosquito finding the exact 2 square centimeters of ankle without bug spray' },
  { slice: 'pets_nature', expectedDomain: 'pets', prompt: 'Squirrel defeating a $120 squirrel-proof bird feeder in twenty minutes' },

  // 14. Everyday Adulting / Domestic Engineering
  { slice: 'adulting', expectedDomain: 'life', prompt: 'Assembling flat-pack Swedish furniture with three leftover wooden dowels' },
  { slice: 'adulting', expectedDomain: 'life', prompt: 'Filing taxes where the government already knows the number but makes you guess' },
  { slice: 'adulting', expectedDomain: 'life', prompt: 'Dentist asking philosophical questions while three instruments are in my mouth' },
  { slice: 'adulting', expectedDomain: 'life', prompt: 'Canceling a gym membership requires a notarized scroll in person' },
  { slice: 'adulting', expectedDomain: 'life', prompt: 'Shampoo and conditioner bottles running out out-of-phase forever' },
  { slice: 'adulting', expectedDomain: 'life', prompt: 'Parallel parking effortlessly when alone vs forgetting geometry when one person watches' },
  { slice: 'adulting', expectedDomain: 'life', prompt: 'Laundry chair reaching geological stratification' },
  { slice: 'adulting', expectedDomain: 'life', prompt: 'Alarm clock stack set for 6:45, 6:50, 6:53, 6:58, and 7:02am' },

  // 15. Internet / Side Projects / Hardware Quirks
  { slice: 'internet_hardware', expectedDomain: 'net_hw', prompt: 'Buying a .dev domain name at 1am and never writing a line of code' },
  { slice: 'internet_hardware', expectedDomain: 'net_hw', prompt: 'Wikipedia click chain going from Medieval pottery to nuclear submarines in 4 minutes' },
  { slice: 'internet_hardware', expectedDomain: 'net_hw', prompt: 'Smart TV taking 18 seconds to boot Linux just to change HDMI input' },
  { slice: 'internet_hardware', expectedDomain: 'net_hw', prompt: 'Forum thread from 2009 ending with never mind fixed it and zero explanation' },
  { slice: 'internet_hardware', expectedDomain: 'net_hw', prompt: 'USB-A plug needing three flips in four-dimensional space to go in' },
  { slice: 'internet_hardware', expectedDomain: 'net_hw', prompt: 'Unsubscribe link asking me to log in to manage communication preferences' },
  { slice: 'internet_hardware', expectedDomain: 'net_hw', prompt: 'Hotel Wi-Fi captive portal intercepting localhost' },
  { slice: 'internet_hardware', expectedDomain: 'net_hw', prompt: 'Autocorrect changing a perfectly spelled word into ducking' },
];

const STOP = new Set([
  'the','a','an','and','or','but','in','on','at','to','for','of','with','by','from','into','over',
  'under','between','after','before','during','while','when','where','why','how','what','who','that',
  'this','these','those','my','our','your','its','their','is','are','was','were','be','been','being',
  'have','has','had','do','does','did','it','i','we','you','they','me','us','them','so','as','if',
  'than','just','only','very','still','every','time','one','two','three','four','five','six','nine',
  'chart','graph','plot','curve','distribution','correlation','flowchart','decision','tree','diagram',
  'workflow','cheat','sheet','vs','versus','trying','making','asking','getting','buying','using',
]);

function stem(w) {
  let s = w.toLowerCase().replace(/[^a-z0-9]/g, '');
  if (s.length > 5 && s.endsWith('ing')) s = s.slice(0, -3);
  else if (s.length > 4 && s.endsWith('ies')) s = s.slice(0, -3) + 'y';
  else if (s.length > 4 && s.endsWith('es') && !s.endsWith('ses')) s = s.slice(0, -1);
  else if (s.length > 3 && s.endsWith('s') && !s.endsWith('ss') && !s.endsWith('us') && !s.endsWith('is')) s = s.slice(0, -1);
  return s;
}

function contentStems(text) {
  return (text.toLowerCase().match(/[a-z0-9]+/g) || [])
    .filter(w => w.length >= 3 && !STOP.has(w))
    .map(stem);
}

function skeletonize(comic, prompt) {
  const promptWords = (prompt.toLowerCase().match(/[a-z0-9]+/g) || []).filter(w => w.length >= 3);
  const lastTurn = comic.turns[comic.turns.length - 1];
  let punch = (lastTurn?.reply ? `${lastTurn.text} // ${lastTurn.reply}` : lastTurn?.text || '') +
    (comic.chart_meta?.callout ? ` | ${comic.chart_meta.callout}` : '') +
    (comic.chart_meta?.no_box ? ` | ${comic.chart_meta.no_box}` : '');
  punch = punch.toUpperCase();
  for (const w of promptWords) {
    const re = new RegExp(`\\b${w.toUpperCase()}S?\\b`, 'g');
    punch = punch.replace(re, '<SLOT>');
  }
  return `${comic.layout}:${punch.replace(/\s+/g, ' ').trim()}`;
}

function checkGrammarAndSlotQuality(comic) {
  const allText = [
    comic.title,
    ...comic.turns.map(t => (t.reply ? `${t.text} ${t.reply}` : t.text)),
    comic.alt,
    comic.chart_meta?.x_label || '',
    comic.chart_meta?.y_label || '',
    comic.chart_meta?.callout || '',
  ].join(' | ');

  const issues = [];
  // 1. Plural noun jammed directly before another plural noun ("TWO BEDSHEETS AGENTS", "TAXES AGENTS", "CATS STANDARDS")
  if (/\b(TWO|THREE|NEW|OUR)\s+[A-Z]{4,}S\s+(AGENTS|STANDARDS|WORKERS|SYSTEMS|PIPELINES|SUPERVISORS)\b/.test(allText.toUpperCase())) {
    issues.push('plural_noun_modifier_jam');
  }
  // 2. Duplicate consecutive words ("AGENT AGENTS", "CHART CHART", "PR PR", "CODE CODE")
  if (/\b([a-z]{3,})\s+\1\b/i.test(allText)) {
    issues.push('duplicate_word_tautology');
  }
  // 3. Broken article ("A APPLE", "A OVERPRICED", "A UNATTACHED") while allowing "A UNIFORM", "A USER", "USB-A AND"
  if (/(?<![A-Za-z]-)\b[Aa]\s+[AEIOUaeiou](?!n\b|ni|se|ti|biq|ne\b|nce\b|ro)[a-zA-Z]{2,}\b/i.test(allText) && /\b[Aa]\s+(?!UNIFORM|USER|UTILITY|UNIQUE|UNIVERSAL|UNION|UNIT|ONE|ONCE|EURO)[AEIOUaeiou][a-zA-Z]{2,}\b/i.test(allText)) {
    issues.push('broken_indefinite_article');
  }
  // 4. Boilerplate non-comic transcript prefixes ("START: EVALUATING", "Y-AXIS: MEASURED THROUGHPUT")
  if (/\b(START:\s*EVALUATING|Y-AXIS:\s*MEASURED THROUGHPUT|INFLECTION:\s*POINT WHERE)\b/.test(allText)) {
    issues.push('dry_boilerplate_transcript');
  }
  // 5. Clipped adjective-only title ("TRAINING NEURAL", "BUYING AN", "ASKING A")
  if (/^(TRAINING|ASKING|WRITING|BUYING|GETTING|DEPLOYING|SPENDING|TRYING|MAKING)\s+[A-Z]+$/.test(comic.title)) {
    issues.push('gerund_clipped_title');
  }
  return issues;
}

function checkSubstringHijack(item, comic) {
  if (item.slice !== 'trap_substrings') return false;
  const full = JSON.stringify(comic).toUpperCase();
  return (
    full.includes('4,200 LINES') ||
    full.includes('TYPE SIGNATURE') ||
    full.includes('60,000 TOKENS/SEC') ||
    full.includes('18.4 GIGABYTES') ||
    full.includes('RELEASE DAG')
  );
}

function scoreWitRubric(item, comic) {
  const turnsText = comic.turns.map(t => (t.reply ? `${t.text} ${t.reply}` : t.text)).join(' ');
  const fullText = `${turnsText} ${comic.alt} ${comic.chart_meta?.callout || ''} ${comic.chart_meta?.no_box || ''}`;
  let score = 0;

  // 1. Semantic grounding (up to 25 pts): incorporates >= 2 distinct stems from prompt (or 100% if prompt has 1 stem)
  const pStems = [...new Set(contentStems(item.prompt))];
  const cStems = new Set(contentStems(`${comic.title} ${fullText}`));
  const hits = pStems.filter(s => cStems.has(s)).length;
  const groundingRatio = pStems.length ? Math.min(1, hits / Math.min(3, pStems.length)) : 1;
  score += groundingRatio * 25;

  // 2. Concrete specificity & quantitative/observational detail (up to 20 pts)
  const hasSpecificDetail = /(\d+|%|AM|PM|MS|SECONDS|MINUTES|HOURS|YEARS|MILES|WATTS|GRAMS|KELVIN|HZ|RFC|ISO|OSHA|IRS|THERMODYNAMIC|ENTROPY|TOPOLOG|BAYES|EQUATION|LAWSUIT|CLOSET|DRAWER|CARDBOARD|MORTGAGE|WARRANTY)/i.test(fullText);
  if (hasSpecificDetail) score += 20;

  // 3. Setup -> Punchline surprise / lexical frame shift (up to 20 pts)
  const firstTurn = comic.turns[0]?.text || '';
  const lastTurn = comic.turns[comic.turns.length - 1];
  const lastText = lastTurn?.reply || lastTurn?.text || '';
  const s1 = new Set(contentStems(firstTurn));
  const s2 = new Set(contentStems(lastText));
  const inter = [...s1].filter(x => s2.has(x)).length;
  const union = new Set([...s1, ...s2]).size || 1;
  const jaccard = inter / union;
  if (jaccard < 0.25 && lastText.length >= 10) score += 20;
  else if (jaccard < 0.45) score += 10;

  // 4. Comic timing / brevity (up to 15 pts): average words per bubble in sweet spot [5..18], punchline <= 16 words
  const wordCounts = comic.turns.map(t => t.text.trim().split(/\s+/).length);
  const avgWords = wordCounts.reduce((a, b) => a + b, 0) / Math.max(1, wordCounts.length);
  const punchWords = lastText.trim().split(/\s+/).length;
  if (avgWords >= 5 && avgWords <= 18 && punchWords <= 17) score += 15;
  else if (avgWords <= 22) score += 8;

  // 5. Independent aftershock alt-text (up to 20 pts): non-overlapping second joke, 8..26 words, zero dry repetition
  const altWords = comic.alt.trim().split(/\s+/).length;
  const altStems = new Set(contentStems(comic.alt));
  const turnStems = new Set(contentStems(turnsText));
  const novelAltStems = [...altStems].filter(x => !turnStems.has(x)).length;
  if (altWords >= 7 && altWords <= 28 && novelAltStems >= 4) score += 20;
  else if (novelAltStems >= 2) score += 10;

  // Penalize grammar/slot flaws or substring hijacks heavily
  const grammarIssues = checkGrammarAndSlotQuality(comic);
  score -= grammarIssues.length * 15;
  if (checkSubstringHijack(item, comic)) score -= 35;

  return Math.max(0, Math.min(100, score));
}

function bootstrapCI(vals, nBoot = 1000, seed = 20261106) {
  let s = seed >>> 0;
  const rand = () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
  const n = vals.length;
  const mean = vals.reduce((a, b) => a + b, 0) / n;
  const means = new Float64Array(nBoot);
  for (let b = 0; b < nBoot; b++) {
    let sum = 0;
    for (let i = 0; i < n; i++) {
      sum += vals[Math.floor(rand() * n)];
    }
    means[b] = sum / n;
  }
  means.sort();
  return {
    mean: Number(mean.toFixed(2)),
    ci95: [Number(means[Math.floor(0.025 * nBoot)].toFixed(2)), Number(means[Math.floor(0.975 * nBoot)].toFixed(2))],
  };
}

export const OOD_PROMPTS = [
  { slice: 'ood_unseen', prompt: 'Hotel blackout curtains always leaving a one-inch laser gap of morning sun' },
  { slice: 'ood_unseen', prompt: 'Folding a giant paper road map in a windy rest stop parking lot' },
  { slice: 'ood_unseen', prompt: 'Carrying twelve heavy grocery bags in one trip to save ninety seconds' },
  { slice: 'ood_unseen', prompt: 'Group chat of six friends trying to pick a dinner restaurant before October' },
  { slice: 'ood_unseen', prompt: 'Choosing a resume typeface for four hours instead of applying to jobs' },
  { slice: 'ood_unseen', prompt: 'Pressing the elevator close-door button eleven times as placebo therapy' },
  { slice: 'ood_unseen', prompt: 'Umbrella inverting inside out the exact millisecond rain turns sideways' },
  { slice: 'ood_unseen', prompt: 'Tupperware cabinet containing nineteen lids and zero matching plastic bowls' },
  { slice: 'ood_unseen', prompt: 'Shopping cart with one rogue caster wheel oscillating at resonant frequency' },
  { slice: 'ood_unseen', prompt: 'Wired earphones tying knots in my pocket via topological braid theory' },
  { slice: 'ood_unseen', prompt: 'Microwave heating the ceramic bowl to 900 degrees while the burrito stays frozen' },
  { slice: 'ood_unseen', prompt: 'Fitted raincoat making me just as wet from internal condensation' },
  { slice: 'ood_unseen', prompt: 'Museum audio guide stuck on track 14 in front of a Renaissance portrait' },
  { slice: 'ood_unseen', prompt: 'Automatic soap dispenser refusing to acknowledge human hands' },
  { slice: 'ood_unseen', prompt: 'Plastic clamshell packaging requiring tin snips to open new scissors' },
  { slice: 'ood_unseen', prompt: 'Escalator etiquette geometry during rush hour commute' },
  { slice: 'ood_unseen', prompt: 'Sticker adhesive residue surviving isopropyl alcohol and sandpaper' },
  { slice: 'ood_unseen', prompt: 'campfires blowing smoke directly at whichever camp chair I move to' },
  { slice: 'ood_unseen', prompt: 'Fitted fitted-sheet corner snapping loose at 4am like a mousetrap' },
  { slice: 'ood_unseen', prompt: 'Popcorn hull wedged between two molars defying physics' },
];

export function evaluateSet(promptItems) {
  const skeletons = new Map();
  const bySliceSkeletons = new Map();
  const grammarClean = [];
  const hijackFree = [];
  const multiConceptRecall = [];
  const witScores = [];
  const latenciesMs = [];

  for (const item of promptItems) {
    const t0 = performance.now();
    const comic = xkcdArtist(item.prompt);
    const svg = comic.toSVG();
    latenciesMs.push(performance.now() - t0);

    if (!svg.includes('<svg')) throw new Error(`Invalid SVG for: ${item.prompt}`);

    const skel = skeletonize(comic, item.prompt);
    skeletons.set(skel, (skeletons.get(skel) || 0) + 1);
    if (!bySliceSkeletons.has(item.slice)) bySliceSkeletons.set(item.slice, []);
    bySliceSkeletons.get(item.slice).push(skel);

    const issues = checkGrammarAndSlotQuality(comic);
    grammarClean.push(issues.length === 0 ? 100 : 0);

    if (item.slice === 'trap_substrings') {
      hijackFree.push(checkSubstringHijack(item, comic) ? 0 : 100);
    }

    const pStems = [...new Set(contentStems(item.prompt))];
    const cStems = new Set(contentStems(`${comic.title} ${JSON.stringify(comic.turns)} ${comic.alt} ${JSON.stringify(comic.chart_meta || {})}`));
    const recalled = pStems.filter(s => cStems.has(s)).length;
    multiConceptRecall.push(pStems.length ? (Math.min(recalled, Math.min(3, pStems.length)) / Math.min(3, pStems.length)) * 100 : 100);

    witScores.push(scoreWitRubric(item, comic));
  }

  let totalPairs = 0;
  let distinctPairs = 0;
  for (const [, list] of bySliceSkeletons) {
    for (let i = 0; i < list.length; i++) {
      for (let j = i + 1; j < list.length; j++) {
        totalPairs++;
        if (list[i] !== list[j]) distinctPairs++;
      }
    }
  }

  latenciesMs.sort((a, b) => a - b);
  return {
    n_prompts: promptItems.length,
    unique_skeletons: skeletons.size,
    structural_uniqueness_pct: Number(((skeletons.size / promptItems.length) * 100).toFixed(2)),
    within_domain_diversity_pct: Number(((distinctPairs / totalPairs) * 100).toFixed(2)),
    substring_trap_accuracy_pct: hijackFree.length ? Number((hijackFree.reduce((a, b) => a + b, 0) / hijackFree.length).toFixed(2)) : null,
    grammar_slot_clean_pct: bootstrapCI(grammarClean),
    semantic_concept_recall_pct: bootstrapCI(multiConceptRecall),
    wit_rubric_score: bootstrapCI(witScores),
    latency_p50_ms: Number(latenciesMs[Math.floor(latenciesMs.length * 0.5)].toFixed(3)),
    latency_p95_ms: Number(latenciesMs[Math.floor(latenciesMs.length * 0.95)].toFixed(3)),
    top_repeated_skeletons: [...skeletons.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([k, count]) => ({ count, skeleton: k.slice(0, 90) })),
  };
}

export function evaluateEngine() {
  return {
    golden_120: evaluateSet(GOLDEN_PROMPTS),
    ood_unseen_20: evaluateSet(OOD_PROMPTS),
  };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const res = evaluateEngine();
  console.log(JSON.stringify(res, null, 2));
  const outArg = process.argv[2];
  if (outArg) {
    fs.writeFileSync(path.resolve(outArg), JSON.stringify(res, null, 2));
  }
}
