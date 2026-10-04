import { readColors, readVar, onThemeChange, type Color } from '../_shared';

export const NAMES = {
  text: '--text', muted: '--text-muted', rule: '--rule', raised: '--bg-raised', sunken: '--bg-sunken',
  accent: '--accent', soft: '--accent-soft', fwd: '--viz-forward', grad: '--viz-gradient', pos: '--viz-pos', neg: '--viz-neg',
  s1: '--viz-seq-1', s2: '--viz-seq-2', s3: '--viz-seq-3', s4: '--viz-seq-4', s5: '--viz-seq-5',
};
export type Palette = Record<keyof typeof NAMES, Color>;

export interface Stage {
  ctx: CanvasRenderingContext2D;
  /** CSS-pixel size of the drawing area. */
  readonly w: number;
  readonly h: number;
  readonly c: Palette;
  mono(px: number, weight?: number): string;
  body(px: number, italic?: boolean): string;
  destroy(): void;
}

/** DPR-aware canvas (cap 2) that follows its parent's size and the site theme. `redraw` is called after any change. */
export function setupStage(cv: HTMLCanvasElement, redraw: () => void): Stage {
  const ctx = cv.getContext('2d')!;
  let w = 1, h = 1, c = readColors(NAMES) as Palette, fonts = { code: readVar('--font-code'), body: readVar('--font-body') };
  const fit = () => {
    const d = Math.min(2, Math.max(1, window.devicePixelRatio || 1)), r = cv.getBoundingClientRect();
    const pw = Math.round(r.width * d), ph = Math.round(r.height * d);
    if (pw && ph && (cv.width !== pw || cv.height !== ph)) { cv.width = pw; cv.height = ph; }
    w = r.width || 1; h = r.height || 1;
    ctx.setTransform(cv.width / w, 0, 0, cv.height / h, 0, 0);
  };
  const ro = new ResizeObserver(() => { fit(); redraw(); });
  ro.observe(cv.parentElement!);
  const off = onThemeChange(() => { c = readColors(NAMES) as Palette; fonts = { code: readVar('--font-code'), body: readVar('--font-body') }; redraw(); });
  void document.fonts.ready.then(() => { fonts = { code: readVar('--font-code'), body: readVar('--font-body') }; redraw(); });
  fit();
  return {
    ctx,
    get w() { return w; }, get h() { return h; }, get c() { return c; },
    mono: (px, wt = 500) => `${wt} ${px}px ${fonts.code}`,
    body: (px, italic = false) => `${italic ? 'italic ' : ''}${px}px ${fonts.body}`,
    destroy() { ro.disconnect(); off(); },
  };
}

/** Number formatting for readouts: fixed decimals with a real minus sign. */
export const f2 = (v: number, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : '∞').replace('-', '−');
