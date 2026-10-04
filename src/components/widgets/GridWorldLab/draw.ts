/** Drawing helpers for the 4x3 grid (no DOM): cells, values, Q triangles, policy arrows, the agent, move outcomes. */
import { COLS, ROWS, DX, DY, GOAL, PIT, START, WALL, NA, isTerminal, xy, type Outcome } from '../../../lib/rl/gridworld';
import { ACCENT, BG, MUTED, RULE, TEXT, arrow, circle, fmt, inkOn, line, mix, poly, rect, signedFill, text, type Box, type Env } from '../_rl/svg';

export interface GridGeom { x0: number; y0: number; cs: number; w: number; h: number }
/** The largest square-celled 4x3 grid that fits in the box, centred. */
export function gridIn(b: Box): GridGeom {
  const cs = Math.floor(Math.min(b.w / COLS, b.h / ROWS));
  return { cs, w: cs * COLS, h: cs * ROWS, x0: b.x + (b.w - cs * COLS) / 2, y0: b.y + (b.h - cs * ROWS) / 2 };
}
/** Top-left corner of cell s, and its centre. */
export function cellXY(G: GridGeom, s: number) {
  const [x, y] = xy(s), px = G.x0 + (x - 1) * G.cs, py = G.y0 + (ROWS - y) * G.cs;
  return { x: px, y: py, cx: px + G.cs / 2, cy: py + G.cs / 2 };
}
/** Pixel centre of a continuous cell position (fx, fy), 1-based like the book (for gliding between cells). */
export const at = (G: GridGeom, fx: number, fy: number): [number, number] => [G.x0 + (fx - 0.5) * G.cs, G.y0 + (ROWS - fy + 0.5) * G.cs];

/** Two grids side by side, or stacked on a phone (each box gets `head` px above it for a caption). */
export function twoBoxes(E: Env, b: Box, gap = 24, head = 22): [Box, Box] {
  if (E.narrow) { const h = (b.h - gap) / 2; return [{ x: b.x, y: b.y + head, w: b.w, h: h - head }, { x: b.x, y: b.y + h + gap + head, w: b.w, h: h - head }]; }
  const w = (b.w - gap) / 2;
  return [{ x: b.x, y: b.y + head, w, h: b.h - head }, { x: b.x + w + gap, y: b.y + head, w, h: b.h - head }];
}

export interface GridOpts {
  /** Fill per cell (non-terminal, non-wall); default the surface. */
  fill?: (s: number) => string | undefined;
  /** Outline one cell with the accent colour. */
  highlight?: number | null;
  /** Show "+1", "−1" and "Start" labels (default true). */
  labels?: boolean;
  /** Tip text per cell (hover). */
  tip?: (s: number) => string;
  /** Click id per cell (needs meta.pickable). */
  pick?: (s: number) => string | undefined;
}
/** Cells, wall, the +1 goal, the −1 pit and the start label. */
export function drawGrid(E: Env, G: GridGeom, o: GridOpts = {}) {
  const lab = o.labels ?? true, big = Math.max(12, Math.min(20, G.cs * 0.22));
  for (let s = 0; s < COLS * ROWS; s++) {
    const c = cellXY(G, s);
    const fill = s === WALL ? mix(MUTED, BG, 45) : s === GOAL ? mix('var(--viz-pos)', BG, 22) : s === PIT ? mix('var(--viz-neg)', BG, 22) : o.fill?.(s) ?? BG;
    rect(E, c.x, c.y, G.cs, G.cs, { fill, stroke: RULE, sw: 1, tip: o.tip?.(s), pick: o.pick?.(s) });
    if (!lab) continue;
    if (s === GOAL) text(E, '+1', c.cx, c.cy + big * 0.35, { size: big, weight: 650, anchor: 'middle', fill: 'var(--viz-pos)' });
    if (s === PIT) text(E, '−1', c.cx, c.cy + big * 0.35, { size: big, weight: 650, anchor: 'middle', fill: 'var(--viz-neg)' });
    if (s === START) text(E, 'Start', c.x + 5, c.y + G.cs - 6, { size: 11, fill: MUTED });
  }
  if (o.highlight != null) { const c = cellXY(G, o.highlight); rect(E, c.x + 1.5, c.y + 1.5, G.cs - 3, G.cs - 3, { fill: 'none', stroke: ACCENT, sw: 2.5 }); }
}

/** Value heatmap with a number per action cell (diverging --viz-pos / --viz-neg around 0). */
export function drawValues(E: Env, G: GridGeom, V: ArrayLike<number>, o: { vmax?: number; digits?: number; show?: (s: number) => number } = {}) {
  const vmax = o.vmax ?? 1, d = o.digits ?? 2, sz = Math.max(11, Math.min(16, G.cs * 0.17));
  for (let s = 0; s < COLS * ROWS; s++) {
    if (s === WALL || isTerminal(s)) continue;
    const a = o.show ? o.show(s) : 1, c = cellXY(G, s), k = Math.min(1, Math.abs(V[s]) / vmax);
    if (a <= 0) continue;
    rect(E, c.x + 0.5, c.y + 0.5, G.cs - 1, G.cs - 1, { fill: signedFill(V[s], vmax), op: a });
    text(E, fmt(V[s], d), c.cx, c.cy + sz * 0.35, { size: sz, anchor: 'middle', weight: 600, fill: inkOn(k), op: a });
  }
}

/** The triangle of action a inside cell s (towards the edge the action points at). */
export function triangle(G: GridGeom, s: number, a: number): [number, number][] {
  const c = cellXY(G, s), { x, y } = c, e = G.cs, m: [number, number] = [c.cx, c.cy];
  return [[[x, y], [x + e, y], m], [[x + e, y], [x + e, y + e], m], [[x + e, y + e], [x, y + e], m], [[x, y + e], [x, y], m]][a] as [number, number][];
}
/** Four Q triangles per action cell, coloured by value; the best action outlined; numbers when the cell is large enough. */
export function drawQ(E: Env, G: GridGeom, Q: ArrayLike<number>, o: { vmax?: number; best?: boolean; numbers?: boolean; digits?: number; show?: (s: number) => number; tip?: (s: number, a: number) => string } = {}) {
  const vmax = o.vmax ?? 1, nums = o.numbers ?? G.cs >= 76, d = o.digits ?? 2;
  for (let s = 0; s < COLS * ROWS; s++) {
    if (s === WALL || isTerminal(s)) continue;
    const a0 = o.show ? o.show(s) : 1;
    if (a0 <= 0) continue;
    let b = 0;
    for (let a = 1; a < NA; a++) if (Q[s * NA + a] > Q[s * NA + b] + 1e-12) b = a;
    const c = cellXY(G, s);
    for (let a = 0; a < NA; a++) {
      const v = Q[s * NA + a];
      poly(E, triangle(G, s, a), { fill: signedFill(v, vmax), stroke: BG, sw: 1, op: a0, tip: o.tip?.(s, a) });
      if (nums) {
        const tx = c.cx + DX[a] * G.cs * 0.3, ty = c.cy - DY[a] * G.cs * 0.3 + 4;
        text(E, fmt(v, d), tx, ty, { size: 11, anchor: 'middle', fill: inkOn(Math.abs(v) / vmax), op: a0 });
      }
    }
    if (o.best !== false) poly(E, triangle(G, s, b), { fill: 'none', stroke: ACCENT, sw: 2.2, op: a0 });
  }
}

/**
 * Policy arrows from each action cell's centre. `probs` (N x 4) draws four arrows whose opacity and width follow
 * the probabilities; `actions` (one per state, -1 for none) draws one arrow per cell.
 */
export function drawPolicy(E: Env, G: GridGeom, o: { actions?: ArrayLike<number>; probs?: ArrayLike<number>; color?: string; flash?: (s: number) => number; show?: (s: number) => number }) {
  const L = G.cs * 0.3, color = o.color ?? TEXT;
  for (let s = 0; s < COLS * ROWS; s++) {
    if (s === WALL || isTerminal(s)) continue;
    const c = cellXY(G, s), vis = o.show ? o.show(s) : 1, f = o.flash?.(s) ?? 0, col = f > 0 ? mix(ACCENT, color, 100 * f) : color;
    if (vis <= 0) continue;
    for (let a = 0; a < NA; a++) {
      const p = o.probs ? o.probs[s * NA + a] : o.actions && o.actions[s] === a ? 1 : 0;
      if (p < 0.02) continue;
      arrow(E, c.cx, c.cy, c.cx + DX[a] * L, c.cy - DY[a] * L, { stroke: col, width: 1 + 2.2 * p, op: vis * (o.probs ? 0.25 + 0.75 * p : 1) });
    }
  }
}

/** The agent: a dot at continuous cell position (fx, fy), moved by a transform. */
export function drawAgent(E: Env, G: GridGeom, fx: number, fy: number, o: { color?: string; r?: number } = {}) {
  const [px, py] = at(G, fx, fy);
  circle(E, 0, 0, o.r ?? Math.max(6, G.cs * 0.12), { fill: o.color ?? ACCENT, stroke: BG, sw: 2, tf: `translate(${px.toFixed(1)} ${py.toFixed(1)})` });
}

/**
 * Outcome arrows of one action from cell s: width ∝ probability, labelled with it. A bump (no move) is drawn as a short
 * arrow that stops at the cell edge with a bar, and the label says "stay".
 */
export function drawOutcomes(E: Env, G: GridGeom, s: number, outs: Outcome[], o: { color?: string; op?: number; labels?: boolean; maxWidth?: number } = {}) {
  const c = cellXY(G, s), color = o.color ?? 'var(--viz-forward)', mw = o.maxWidth ?? 9;
  for (const out of outs) {
    const dx = DX[out.dir], dy = -DY[out.dir], w = 1.2 + mw * out.p;
    if (out.moved) {
      const t = cellXY(G, out.s2);
      arrow(E, c.cx + dx * G.cs * 0.18, c.cy + dy * G.cs * 0.18, t.cx - dx * G.cs * 0.16, t.cy - dy * G.cs * 0.16, { stroke: color, width: w, op: o.op });
    } else {
      const ex = c.cx + dx * G.cs * 0.46, ey = c.cy + dy * G.cs * 0.46;
      arrow(E, c.cx + dx * G.cs * 0.18, c.cy + dy * G.cs * 0.18, ex, ey, { stroke: color, width: w, op: o.op });
      line(E, ex - dy * G.cs * 0.16, ey - dx * G.cs * 0.16, ex + dy * G.cs * 0.16, ey + dx * G.cs * 0.16, { stroke: TEXT, sw: 2.5, op: o.op });
    }
    if (o.labels !== false) {
      const lx = c.cx + dx * G.cs * (out.moved ? 0.62 : 0.3) + (dy ? 14 : 0), ly = c.cy + dy * G.cs * (out.moved ? 0.62 : 0.3) + (dx ? -8 : 4);
      text(E, (out.moved ? '' : 'stay ') + fmt(out.p, 1), lx, ly, { size: 12, anchor: 'middle', weight: 600, halo: true, op: o.op });
    }
  }
}
