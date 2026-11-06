/**
 * Level 5 Agent-Native & WebMCP (`navigator.modelContext`) interface for xkcd-artist Studio.
 * Exposes `window.xkcdArtistStudio` and registers machine-callable tools for browser AI agents.
 */
import { CHARACTERS, HF_MODEL_REPO, LAYOUT_TYPES, PROPS } from './index.js';

export function registerXkcdAgentCapabilities({
  deck,
  getCurrentComic,
  synthesizePrompt,
  getEmbeddedSvg,
}) {
  const synthesizeTool = async ({ prompt } = {}) => {
    const q = String(prompt || '').trim();
    if (!q) {
      return { error: 'Provide a non-empty engineering, science, chart, or flowchart prompt.' };
    }
    const comic = await synthesizePrompt(q, true);
    return {
      title: comic.title,
      prompt: comic.prompt,
      layout: comic.layout,
      prop: comic.prop,
      cast: comic.cast,
      turns: comic.turns,
      alt: comic.alt,
      style: comic.style,
      tier: comic.tier,
      confidence: comic.confidence,
      p_invalid: comic.p_invalid,
      neural: comic.neural || null,
    };
  };

  const listPresetsTool = ({ layout } = {}) => {
    const filter = layout ? String(layout).trim().toLowerCase() : '';
    const items = deck.map((d, index) => ({ index, label: d.label, prompt: d.prompt, layout: d.layout }));
    const filtered = filter ? items.filter(i => i.layout === filter) : items;
    return {
      total: deck.length,
      count: filtered.length,
      layouts_supported: LAYOUT_TYPES,
      characters_supported: CHARACTERS,
      props_supported: PROPS,
      presets: filtered,
    };
  };

  const getTelemetryTool = () => {
    const comic = getCurrentComic();
    if (!comic) return { status: 'initializing' };
    return {
      model_repo: HF_MODEL_REPO,
      tier: comic.tier,
      title: comic.title,
      layout: comic.layout,
      prop: comic.prop,
      cast: comic.cast,
      style: comic.style,
      confidence: comic.confidence,
      p_invalid: comic.p_invalid,
      neural: comic.neural || null,
    };
  };

  const exportSvgTool = async ({ embedFont = false } = {}) => {
    const svg = await getEmbeddedSvg(Boolean(embedFont));
    const comic = getCurrentComic();
    return {
      title: comic?.title || 'XKCD',
      alt: comic?.alt || '',
      svg_length: svg.length,
      svg,
    };
  };

  const api = {
    synthesizeComic: synthesizeTool,
    listPresets: listPresetsTool,
    getModelTelemetry: getTelemetryTool,
    exportSvg: exportSvgTool,
    modelRepo: HF_MODEL_REPO,
  };

  if (typeof window !== 'undefined') {
    window.xkcdArtistStudio = api;
    window.__xkcdArtist = api;
  }

  const mc = typeof navigator !== 'undefined' ? navigator.modelContext : null;
  if (mc && typeof mc.registerTool === 'function') {
    const tools = [
      {
        name: 'synthesize_xkcd_comic',
        description:
          'Synthesize an authentic monochrome XKCD comic strip, annotated chart, or decision flowchart from any text prompt using the gnumanth/xkcd-artist ONNX Int8 WASM planner and xkcd-script hand-lettering.',
        schema: {
          type: 'object',
          properties: {
            prompt: {
              type: 'string',
              description: 'Scenario, chart, or flowchart prompt (e.g. "Flashing a 1.7MB WebAssembly neural model onto a smart toaster" or "Annotated quantization curve from FP32 down to 0.5 bits per weight").',
            },
          },
          required: ['prompt'],
        },
        run: synthesizeTool,
      },
      {
        name: 'list_xkcd_presets',
        description:
          'List all curated novel and classic XKCD comic presets across single_panel, two_panel, three_panel, four_panel, annotated_chart, and flowchart_diagram layouts.',
        schema: {
          type: 'object',
          properties: {
            layout: {
              type: 'string',
              enum: LAYOUT_TYPES,
              description: 'Optional layout filter.',
            },
          },
        },
        run: async args => listPresetsTool(args),
      },
      {
        name: 'get_xkcd_model_telemetry',
        description:
          'Inspect live ONNX Runtime WebAssembly telemetry from gnumanth/xkcd-artist (predicted layout/prop confidence, stroke style vector, 12-DOF poses, abstention p_invalid, and inference latency).',
        schema: {
          type: 'object',
          properties: {},
        },
        run: async () => getTelemetryTool(),
      },
      {
        name: 'export_xkcd_svg',
        description:
          'Export the currently rendered XKCD comic as standalone SVG markup (optionally embedding base64 xkcd-script.ttf).',
        schema: {
          type: 'object',
          properties: {
            embedFont: {
              type: 'boolean',
              description: 'Embed base64 xkcd-script.ttf @font-face inside the returned SVG.',
            },
          },
        },
        run: exportSvgTool,
      },
    ];

    for (const t of tools) {
      try {
        mc.registerTool({
          name: t.name,
          description: t.description,
          inputSchema: t.schema,
          parameters: t.schema,
          execute: t.run,
          handler: t.run,
        });
      } catch {
        // Ignore experimental WebMCP signature differences across browsers
      }
    }
  }

  return api;
}
