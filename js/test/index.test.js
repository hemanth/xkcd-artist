import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import xkcdArtist, {
  LAYOUT_TYPES,
  CHARACTERS,
  PROPS,
  encodePromptTokens,
  HF_MODEL_REPO,
} from '../src/index.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..', '..');

function assertValidXkcdSvg(svg, expectedSubstrings = []) {
  const markupWithoutBase64 = svg.replace(/data:(?:image\/png|font\/ttf);base64,[A-Za-z0-9+/=]+/g, 'DATA_URI');
  assert.ok(svg.startsWith('<svg '), 'SVG must start with <svg');
  assert.ok(svg.endsWith('</svg>'), 'SVG must end with </svg>');
  assert.ok(!markupWithoutBase64.includes('NaN'), 'SVG must contain zero NaN coordinates');
  assert.ok(!markupWithoutBase64.includes('undefined'), 'SVG must contain zero undefined values');
  assert.ok(!svg.includes('#f7f6f0'), 'Default comic SVG must be pure monochrome without beige alt chrome');
  assert.ok(svg.includes('<title>'), 'SVG must include <title> for hover alt-text');
  assert.ok(svg.includes('<desc>'), 'SVG must include <desc>');

  const openGroups = (svg.match(/<g\b/g) || []).length;
  const closeGroups = (svg.match(/<\/g>/g) || []).length;
  assert.equal(openGroups, closeGroups, 'Balanced <g> tags');

  for (const sub of expectedSubstrings) {
    assert.ok(svg.includes(sub), `Expected SVG to include "${sub}"`);
  }
}

test('xkcdArtist synthesizes all 6 iconic Randall Munroe classics with pure monochrome SVG', () => {
  const classics = [
    ['Sudo make me a sandwich', 'SANDWICH', 'single_panel', ['SUDO MAKE ME A SANDWICH', 'Simon Says']],
    ['Compiling code sword fight', 'COMPILING', 'single_panel', ['COMPILING!', 'LCDs']],
    ['Machine learning pile of linear algebra', 'MACHINE LEARNING', 'single_panel', ['LINEAR ALGEBRA', 'recurrent']],
    ['Modern digital infrastructure maintained in Nebraska', 'DEPENDENCY', 'single_panel', ['NEBRASKA', 'ImageMagick']],
    ['14 competing standards', 'STANDARDS', 'three_panel', ['HOW STANDARDS PROLIFERATE:', '15 COMPETING', 'YEAH!']],
    ['Someone is wrong on the internet', 'DUTY CALLS', 'single_panel', ['WRONG ON', 'THE INTERNET.']],
  ];

  for (const [prompt, title, layout, expected] of classics) {
    const comic = xkcdArtist(prompt);
    assert.equal(comic.is_invalid, false);
    assert.equal(comic.title, title);
    assert.equal(comic.layout, layout);
    assert.ok(comic.panels.length >= 2);
    assertValidXkcdSvg(comic.toSVG(), expected);
  }
});

test('xkcdArtist covers all 6 LAYOUT_TYPES across strips, charts, and flowcharts', () => {
  const promptsByLayout = {
    single_panel: 'Git rebase before release',
    two_panel: 'Deploying a config tweak on Friday at 5pm',
    three_panel: 'Why my Docker container image is 18GB',
    four_panel: 'Reviewing a 4200 line pull request in TypeScript',
    annotated_chart: 'Annotated chart of time spent tuning webpack config',
    flowchart_diagram: 'Flowchart for deciding whether to touch legacy DNS',
  };

  const seen = new Set();
  for (const [expectedLayout, prompt] of Object.entries(promptsByLayout)) {
    const comic = xkcdArtist(prompt);
    assert.equal(comic.layout, expectedLayout, `Prompt "${prompt}" should produce ${expectedLayout}`);
    assert.ok(CHARACTERS.includes(comic.cast[0]));
    assert.ok(PROPS.includes(comic.prop));
    const svg = comic.toSVG();
    assertValidXkcdSvg(svg, ['<path']);
    seen.add(comic.layout);
  }
  assert.deepEqual([...seen].sort(), [...LAYOUT_TYPES].sort());
});

test('xkcdArtist prevents keyword tautologies on developer prompts', () => {
  const pr = xkcdArtist('Reviewing a 4200 line pull request in TypeScript');
  const prSvg = pr.toSVG();
  assert.ok(!prSvg.includes('PULL PR'));
  assert.ok(prSvg.includes('TYPESCRIPT PR'));

  const fri = xkcdArtist('Deploying a config tweak on Friday at 5pm');
  assert.ok(!fri.toSVG().includes('CONFIG CONFIG'));

  const dk = xkcdArtist('Why my Docker container image is 18GB');
  assert.ok(!dk.toSVG().includes('18GB IMAGE'));
  assert.ok(dk.toSVG().includes('DOCKER IMAGE'));
});

test('toSVG supports explicit dimensions, showChrome, and embedded @font-face CSS', () => {
  const comic = xkcdArtist('Autonomous AI agents automating my workflow');
  const sized = comic.toSVG(920, 400);
  assert.ok(sized.includes('width="920" height="400"'));
  assert.ok(sized.includes('preserveAspectRatio="xMidYMid meet"'));

  const chromed = comic.toSVG({ showChrome: true, embedFontCss: '@font-face{font-family:xkcd-script;}' });
  assert.ok(chromed.includes('Title text:'));
  assert.ok(chromed.includes('<style>@font-face{font-family:xkcd-script;}</style>'));
});

test('xkcdArtist handles arbitrary general prompts and abstains on empty/garbage inputs', () => {
  const general = xkcdArtist('Modeling quantum espresso thermodynamics from first principles');
  assert.equal(general.is_invalid, false);
  assert.ok(general.confidence > 0.99);
  assertValidXkcdSvg(general.toSVG());

  for (const bad of [
    '',
    '   ',
    'bad\0input',
    '404 NOT FOUND EMPTY PROMPT ABSTENTION',
    'SELECT * FROM comics WHERE id = -99999',
    'null undefined NaN [[[',
  ]) {
    const invalid = xkcdArtist(bad);
    assert.equal(invalid.is_invalid, true, `Expected abstention for ${JSON.stringify(bad)}`);
    assert.ok(invalid.p_invalid > 0.99);
  }
});

test('js/src/index.js and docs/src/index.js stay byte-for-byte identical', () => {
  const pkgSrc = fs.readFileSync(path.join(ROOT, 'js', 'src', 'index.js'), 'utf8');
  const docsSrc = fs.readFileSync(path.join(ROOT, 'docs', 'src', 'index.js'), 'utf8');
  assert.equal(pkgSrc, docsSrc);

  const pkgOnnx = fs.readFileSync(path.join(ROOT, 'js', 'src', 'onnx_planner.js'), 'utf8');
  const docsOnnx = fs.readFileSync(path.join(ROOT, 'docs', 'src', 'onnx_planner.js'), 'utf8');
  assert.equal(pkgOnnx, docsOnnx);

  const pkgKin = fs.readFileSync(path.join(ROOT, 'js', 'src', 'kinematic_renderer.js'), 'utf8');
  const docsKin = fs.readFileSync(path.join(ROOT, 'docs', 'src', 'kinematic_renderer.js'), 'utf8');
  assert.equal(pkgKin, docsKin);
});

test('synthesizes 12 brand-new non-XKCD original comics as 100% pure procedural SVG modulated by ONNX 12-DOF poses', () => {
  const uniquePrompts = [
    ['Running a quantized neural network inside a smart toaster via WebAssembly', 'three_panel'],
    ['Giving two browser MCP agents write access to my calendar and inbox', 'four_panel'],
    ['Calculating H100 VRAM rent per square millimeter for a 2M token KV cache', 'two_panel'],
    ['Quantization curve from FP32 down to 0.5 bits per weight', 'annotated_chart'],
    ['Flowchart for deciding whether a model belongs in browser WebAssembly', 'flowchart_diagram'],
    ['Security auditing a production SaaS app vibe-coded in fourteen minutes', 'three_panel'],
    ['Adding Mars Sol time UTC+24h39m35s to POSIX tzdata', 'single_panel'],
    ['Bio-engineering a sourdough starter that ferments in O(log n) time', 'three_panel'],
    ['Bimodal capability chart of identical matte black USB-C cables in my desk drawer', 'annotated_chart'],
    ['Three cloud vendors implementing an IETF RFC number an LLM made up', 'two_panel'],
    ['Robot vacuum LiDAR SLAM map classifying a sleeping cat as a moving tectonic plate', 'single_panel'],
    ['Thermostat flowchart for heating an apartment with local LoRA fine-tuning', 'flowchart_diagram'],
  ];

  for (const [prompt, expectedLayout] of uniquePrompts) {
    const comic = xkcdArtist(prompt);
    assert.equal(comic.scene, 'kinematic');
    assert.equal(comic.layout, expectedLayout);
    const svg = comic.toSVG();
    assertValidXkcdSvg(svg, ['<path']);
    assert.ok(!svg.includes('<image '), `Novel comic "${prompt}" must be 100% pure vector SVG with zero raster <image> crops`);
  }

  const base = xkcdArtist('Running a quantized neural network inside a smart toaster via WebAssembly');
  const svgA = base.toSVG();
  base.applyNeural({
    model_id: 'gnumanth/xkcd-artist',
    latency_ms: 0.52,
    style: { wobble_amp_px: 1.92, stroke_width_px: 2.18, bubble_zone_ratio: 0.28, ink_contrast: 0.92 },
    poses_12dof: [0.28, 0.85, 0.32, -0.4, 0.75, 1.0, 0.72, 0.85, -0.28, 0.5, -0.65, 1.0],
    p_invalid: 0.0002,
  });
  const svgB = base.toSVG();
  assert.notEqual(svgA, svgB, 'ONNX 12-DOF pose vector + style_vec must dynamically modulate stick-figure SVG geometry');
});

test('encodePromptTokens matches Python MD5 vocabulary encoding and applyNeural modulates SVG ink', () => {
  assert.equal(HF_MODEL_REPO, 'https://huggingface.co/gnumanth/xkcd-artist');
  const ids = encodePromptTokens('turns_3 three_panel laptop_desk DOCKER CONTAINER Why my Docker container image is 18GB', 36);
  assert.equal(ids.length, 36);
  assert.deepEqual(ids.slice(0, 12), [186, 801, 1111, 21, 1184, 1198, 161, 21, 1184, 958, 116, 690]);

  const comic = xkcdArtist('Why my Docker container image is 18GB');
  const defaultSvg = comic.toSVG();
  comic.applyNeural({
    model_id: 'gnumanth/xkcd-artist',
    latency_ms: 0.55,
    style: { wobble_amp_px: 2.15, stroke_width_px: 2.45, bubble_zone_ratio: 0.27, ink_contrast: 0.93 },
    p_invalid: 0.0004,
  });
  assert.equal(comic.tier, 'onnx_int8_wasm_live');
  assert.equal(comic.confidence, 0.9996);
  const neuralSvg = comic.toSVG();
  assert.notEqual(defaultSvg, neuralSvg, 'Neural style_vec should modulate SVG wobbled path geometry & stroke width');
});
