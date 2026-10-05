/**
 * mdp (rl-basics step 2): the transition probabilities P(s' | s, a) of one cell. The timeline cycles through the four
 * actions; for each, the outcome arrows (width = exact probability) fade in, then 20 seeded tries drop into their
 * landing cells and a tally fills next to the exact chances.
 * Props: focus ("3,1"), seed (5), slip (0.2).
 */
import { createRng } from '../../../lib/math/rng';
import { ACTIONS, ARROWS, DX, DY, cellName, step, transitions, xy, type Grid, type Outcome } from '../../../lib/rl/gridworld';
import { int, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { MUTED, circle, ease, fmt, lerp, text, type Env } from '../_rl/svg';
import { focusOf, gridOf } from './sim';
import { at, cellXY, drawGrid, drawOutcomes, gridIn } from './draw';

const SEG = 5.4, FADE = 1.0, DOT0 = 1.2, GAP = 0.12, FLY = 0.5, TRIES = 20;
interface Group { s2: number; p: number }
interface Scene { g: Grid; focus: number; outs: Outcome[][]; groups: Group[][]; samples: number[][] }
interface F extends Frame { a: number; landed: number; arrowOp: number; flying: { i: number; u: number }[]; tally: Map<number, number> }

/** Outcomes with the same landing cell merged (two bumps both mean "stay"), biggest first. */
const group = (outs: Outcome[]): Group[] => {
  const m = new Map<number, number>();
  for (const o of outs) m.set(o.s2, (m.get(o.s2) ?? 0) + o.p);
  return [...m].map(([s2, p]) => ({ s2, p })).sort((x, y) => y.p - x.p || x.s2 - y.s2);
};
const name = (s: number, focus: number) => (s === focus ? 'stay put' : cellName(s));

const mode: ModeImpl<Scene, F> = {
  setup(o: Opts) {
    const g = gridOf(o), focus = focusOf(o, 2), seed = int(o.seed, 5);
    const outs = [0, 1, 2, 3].map((a) => transitions(g, focus, a));
    const samples = [0, 1, 2, 3].map((a) => { const rng = createRng(seed * 100 + a); return Array.from({ length: TRIES }, () => step(g, focus, a, rng).s2); });
    return { g, focus, outs, groups: outs.map(group), samples };
  },
  timeline: () => ({ duration: SEG * 4, markers: ACTIONS.map((a, i) => ({ at: i * SEG, label: `Ask ${a}` })) }),
  frameAt(t, s) {
    const a = Math.max(0, Math.min(3, Math.floor(t / SEG + 1e-9))), u = Math.min(SEG, t - a * SEG);
    const landed = Math.max(0, Math.min(TRIES, Math.floor((u - DOT0 - FLY) / GAP) + 1));
    const flying = [];
    for (let i = landed; i < TRIES; i++) { const w = (u - DOT0 - i * GAP) / FLY; if (w <= 0) break; flying.push({ i, u: w }); }
    const tally = new Map<number, number>();
    for (let i = 0; i < landed; i++) tally.set(s.samples[a][i], (tally.get(s.samples[a][i]) ?? 0) + 1);
    const exact = s.groups[a].map((x) => `${fmt(x.p, 1)} to ${name(x.s2, s.focus)}`).join(', ');
    const got = s.groups[a].filter((x) => tally.get(x.s2)).map((x) => `${tally.get(x.s2)} to ${name(x.s2, s.focus)}`).join(', ');
    const who = `asking for ${ACTIONS[a]} from ${cellName(s.focus)}`;
    const status = landed === 0 ? `The agent is ${who}. The exact chances: ${exact}.`
      : landed < TRIES ? `${landed} of ${TRIES} tries ${who} so far: ${got}.`
      : `All ${TRIES} tries ${who}: ${got}. The exact chances were ${exact}.`;
    return { a, landed, arrowOp: ease((u - 0.2) / (FADE - 0.2)), flying, tally, status, readouts: [['Asking for', `${ACTIONS[a]} ${ARROWS[a]}`], ['Tries', `${landed} of ${TRIES}`]] };
  },
  draw(E: Env, f, s) {
    const nar = E.narrow, gb = nar ? { x: 8, y: 8, w: E.W - 16, h: E.H * 0.5 } : { x: 8, y: 8, w: E.W * 0.58, h: E.H - 16 };
    const G = gridIn(gb);
    drawGrid(E, G, { highlight: s.focus, labels: true });
    drawOutcomes(E, G, s.focus, s.outs[f.a], { op: f.arrowOp, labels: false });
    // labels drawn here so 'stay' sits in the cell's top-left corner, clear of the bump arrow and the sample dots
    const fc = cellXY(G, s.focus);
    for (const out of s.outs[f.a]) if (out.moved) text(E, fmt(out.p, 1), fc.cx + DX[out.dir] * G.cs * 0.62 + (DY[out.dir] ? 14 : 0), fc.cy - DY[out.dir] * G.cs * 0.62 + (DX[out.dir] ? -8 : 4), { size: 12, anchor: 'middle', weight: 600, halo: true, op: f.arrowOp });
    const stay = s.outs[f.a].filter((x) => !x.moved).reduce((x, y) => x + y.p, 0);
    if (stay > 0) text(E, `stay ${fmt(stay, 1)}`, fc.x + 6, fc.y + 15, { size: 12, weight: 600, halo: true, op: f.arrowOp });
    // slot k of a cell: a small lattice along the cell's lower edge
    const slot = (cell: number, k: number): [number, number] => { const c = cellXY(G, cell); return [c.x + 11 + (k % 6) * 10, c.y + G.cs - 12 - Math.floor(k / 6) * 10]; };
    const used = new Map<number, number>(), at0 = at(G, xy(s.focus)[0], xy(s.focus)[1]);
    for (let i = 0; i < f.landed; i++) {
      const cell = s.samples[f.a][i], k = used.get(cell) ?? 0; used.set(cell, k + 1);
      const [x, y] = slot(cell, k);
      circle(E, x, y, 4.2, { fill: 'var(--viz-forward)', stroke: 'var(--bg-raised)', sw: 1 });
    }
    for (const d of f.flying) {
      const cell = s.samples[f.a][d.i], k = [...Array(d.i).keys()].filter((j) => s.samples[f.a][j] === cell).length, [x1, y1] = slot(cell, k), m = ease(d.u);
      circle(E, lerp(at0[0], x1, m), lerp(at0[1], y1, m) - 22 * Math.sin(Math.PI * m), 4.6, { fill: 'var(--viz-forward)', stroke: 'var(--bg-raised)', sw: 1 });
    }
    // exact chance against the tally
    const tx = nar ? 12 : G.x0 + G.w + 26, ty = nar ? gb.y + gb.h + 30 : 34, w = nar ? E.W - 24 : E.W - tx - 10, sz = 12;
    text(E, `Asking for ${ACTIONS[f.a]} ${ARROWS[f.a]} from ${cellName(s.focus)}`, tx, ty, { weight: 650, size: 13 });
    const cols = [0, w * 0.46, w * 0.72];
    ['Lands in', 'Exact', 'Tries so far'].forEach((h, j) => text(E, h, tx + cols[j], ty + 26, { size: sz, fill: MUTED }));
    s.groups[f.a].forEach((x, i) => {
      const y = ty + 48 + i * 22;
      text(E, name(x.s2, s.focus), tx + cols[0], y, { size: sz });
      text(E, fmt(x.p, 1), tx + cols[1], y, { size: sz, mono: true });
      text(E, String(f.tally.get(x.s2) ?? 0), tx + cols[2], y, { size: sz, mono: true });
    });
  },
  label: (s) => `A cell of the grid with arrows showing where each of the four moves can end up; twenty sampled tries per move fall into the cells around ${cellName(s.focus)}.`,
  restTime: () => SEG - 0.4,
};
export default mode;
