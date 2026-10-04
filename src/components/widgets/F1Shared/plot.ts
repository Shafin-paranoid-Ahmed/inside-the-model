/** Pure helpers shared by the Day 1 widgets: marching-squares contours, panel layout, easing. No DOM. */

/** Marching squares: line segments [x1,y1,x2,y2,...] (in grid-index units) where v(i,j) crosses `level`. */
export function contour(v: (i: number, j: number) => number, nx: number, ny: number, level: number): number[] {
  const out: number[] = [];
  const seg = (p: number[], q: number[]) => out.push(p[0], p[1], q[0], q[1]);
  for (let j = 0; j < ny - 1; j++) {
    for (let i = 0; i < nx - 1; i++) {
      const a = v(i, j), b = v(i + 1, j), c = v(i + 1, j + 1), d = v(i, j + 1);
      const k = (a > level ? 1 : 0) | (b > level ? 2 : 0) | (c > level ? 4 : 0) | (d > level ? 8 : 0);
      if (k === 0 || k === 15) continue;
      const f = (p: number, q: number) => (level - p) / (q - p);
      const T = [i + f(a, b), j], R = [i + 1, j + f(b, c)], B = [i + f(d, c), j + 1], L = [i, j + f(a, d)];
      switch (k) {
        case 1: case 14: seg(L, T); break;
        case 2: case 13: seg(T, R); break;
        case 3: case 12: seg(L, R); break;
        case 4: case 11: seg(R, B); break;
        case 5: seg(L, T); seg(R, B); break;
        case 6: case 9: seg(T, B); break;
        case 7: case 8: seg(L, B); break;
        case 10: seg(T, R); seg(L, B); break;
      }
    }
  }
  return out;
}

export interface Box { x: number; y: number; w: number; h: number }

/** Split a W×H stage into a main panel and a side panel: side by side when wide, stacked when narrow. */
export function splitPanels(W: number, H: number, mainFrac = 0.6): { main: Box; side: Box; stacked: boolean } {
  const stacked = W < 540;
  const g = 10;
  if (stacked) {
    const mh = Math.round(H * mainFrac);
    return { stacked, main: { x: 0, y: 0, w: W, h: mh - g / 2 }, side: { x: 0, y: mh + g / 2, w: W, h: H - mh - g / 2 } };
  }
  const mw = Math.round(W * mainFrac);
  return { stacked, main: { x: 0, y: 0, w: mw - g / 2, h: H }, side: { x: mw + g / 2, y: 0, w: W - mw - g / 2, h: H } };
}

/** Linear map from a data range to a pixel range. */
export const scale = (d0: number, d1: number, p0: number, p1: number) => (v: number) => p0 + ((v - d0) / (d1 - d0)) * (p1 - p0);

export const smooth = (u: number) => { const c = Math.min(1, Math.max(0, u)); return c * c * (3 - 2 * c); };
export const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));
