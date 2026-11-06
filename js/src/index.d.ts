export interface XkcdTurn {
  speaker: string;
  text: string;
  reply?: string;
}

export interface XkcdPanel {
  index: number;
  cast: string[];
  prop: string;
  turns: XkcdTurn[];
}

export interface XkcdSvgOptions {
  width?: number;
  height?: number;
  showChrome?: boolean;
  embedFontCss?: string;
}

export interface XkcdComicResult {
  prompt: string;
  title: string;
  scene: string;
  header: string;
  alt: string;
  layout: string;
  prop: string;
  cast: string[];
  panels: XkcdPanel[];
  style: {
    wobble_amp_px: number;
    stroke_width_px: number;
    bubble_zone_ratio: number;
    ink_contrast: number;
  };
  p_invalid: number;
  is_invalid: boolean;
  confidence: number;
  tier: string;
  neural?: Record<string, unknown> | null;
  applyNeural(neuralOut: Record<string, unknown>): XkcdComicResult;
  toSVG(optionsOrWidth?: number | XkcdSvgOptions, height?: number): string;
}

export const LAYOUT_TYPES: string[];
export const CHARACTERS: string[];
export const PROPS: string[];
export const HF_MODEL_REPO: string;
export const HF_ONNX_URL: string;
export function encodePromptTokens(text: string, maxLen?: number): number[];
export function loadOnnxPlanner(options?: { modelUrls?: string[]; ortModule?: unknown }): Promise<unknown>;
export function runOnnxPlanner(
  enrichedPrompt: string,
  cast?: string[],
  options?: { modelUrls?: string[]; ortModule?: unknown }
): Promise<Record<string, unknown>>;
export function xkcdArtistAsync(
  prompt: string | { prompt?: string; title?: string },
  options?: { panels?: number; wobbleAmp?: number; strokeWidth?: number; tier?: string; modelUrls?: string[] }
): Promise<XkcdComicResult>;
export function renderXkcdSVG(
  spec: Record<string, unknown>,
  optionsOrWidth?: number | XkcdSvgOptions,
  maybeHeight?: number
): string;

export default function xkcdArtist(
  prompt: string | { prompt?: string; title?: string },
  options?: { panels?: number; wobbleAmp?: number; strokeWidth?: number; tier?: string }
): XkcdComicResult;
