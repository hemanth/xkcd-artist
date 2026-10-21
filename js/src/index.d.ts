export interface XkcdTurn {
  speaker: string;
  text: string;
}

export interface XkcdPanel {
  index: number;
  cast: string[];
  prop: string;
  turns: XkcdTurn[];
}

export interface XkcdComicResult {
  prompt: string;
  title: string;
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
  toSVG(width?: number, height?: number): string;
}

export default function xkcdArtist(
  prompt: string | { prompt?: string; title?: string },
  options?: { panels?: number; wobbleAmp?: number; strokeWidth?: number }
): XkcdComicResult;
