/**
 * dqn-target (deep-rl step 5): 300 updates of DQN with replay (capacity 256, batch 16) and a target network copied
 * every 50 updates (Adam 0.01, seed 19). One fixed diagnostic transition, (3,1) up, shows the online prediction Q and
 * the target y after every update.
 * Measured (seed 19): y 0.321 after updates 1-49 (here 48 and 49), 0.131 from update 50 to 99; weight distance 1.488 at
 * update 49, 0 at update 50, 0.024 at update 51. Q* is 0.397.
 * Props: seed (19), updates (300), copyEvery (50), lr (0.01), capacity (256), batch (16).
 */
import { ACTIONS, cellName } from '../../../lib/rl/gridworld';
import { optimal } from '../../../lib/rl/tabular';
import { type DQNResult } from '../../../lib/rl/dqn';
import { int, num, type Frame, type Kit, type ModeImpl, type Opts } from '../_rl/kit';
import { MUTED, RULE, TEXT, arrow, axes, fmt, line, path, poly, rect, text, type Env } from '../_rl/svg';
import { DIAG, GAMMA, dqnGrid } from './dqn-common';

const HOLD = 2;
interface Scene { r: DQNResult; n: number; copy: number; qstar: number; dmax: number; knots: [number, number][] }
interface F extends Frame { k: number }
const kAt = (t: number, s: Scene) => {
  const K = s.knots;
  for (let i = 1; i < K.length; i++) if (t <= K[i][0]) { const [t0, k0] = K[i - 1], [t1, k1] = K[i]; return Math.max(1, Math.round(k0 + ((t - t0) / (t1 - t0)) * (k1 - k0))); }
  return s.n;
};

const mode: ModeImpl<Scene, F> = {
  async setup(o: Opts, kit: Kit) {
    const g = dqnGrid(o), n = int(o.updates, 300), copy = int(o.copyEvery ?? o.targetEvery, 50);
    const r = await kit.job<DQNResult>('trainDQN', g, { optimizer: 'adam', lr: num(o.lr, 0.01), gamma: GAMMA, hidden: int(o.hidden, 16), replay: true, target: true, capacity: int(o.capacity, 256), batch: int(o.batch, 16), copyEvery: copy, seed: int(o.seed, 19), diag: DIAG, updates: n, evalEvery: 100000 });
    const k0 = copy - 5, k1 = copy + 5;
    return { r, n: r.updates, copy, qstar: optimal(g, GAMMA).Q[DIAG.s * 4 + DIAG.a], dmax: Math.max(...Array.from(r.weightDist)), knots: [[0, 0], [2, k0], [5, k1], [11, r.updates]] };
  },
  timeline: (s) => ({ duration: s.knots[3][0] + HOLD, markers: [{ at: 0, label: `Updates 1–${s.copy - 5}` }, { at: 2, label: `Around the copy at ${s.copy}` }, { at: 5, label: `Updates ${s.copy + 5}–${s.n}` }] }),
  frameAt(t, s) {
    const k = kAt(t, s), i = k - 1, { r } = s, q = r.diagQ[i], y = r.diagY[i], d = r.weightDist[i], last = Math.floor(k / s.copy) * s.copy;
    const status = k % s.copy === 0
      ? `Update ${k}: the online weights were just copied into the target network, so the distance between them is 0 and the target for the next update changes from ${fmt(r.diagY[i - 1])} to ${fmt(y)}.`
      : `After update ${k} the online network predicts ${fmt(q)} for this move. The target network has not changed ${last ? `since update ${last}` : 'since the start'}, so the target is still ${fmt(y)}; the weights have drifted ${fmt(d, 3)} apart.`;
    return { k, status, readouts: [['Update', `${k} of ${s.n}`], ['Online Q', fmt(q)], ['Target y', fmt(y)], ['Weight distance', fmt(d, 3)]] };
  },
  draw(E: Env, f, s) {
    const nar = E.narrow, { r } = s, i = f.k - 1, band = nar ? 96 : 104, bw = nar ? (E.W - 16 - 56) / 2 : Math.min(230, (E.W - 40 - 120) / 2);
    const boxes = [{ x: 8, y: 8, w: bw, h: band - 16, t: 'Online network', sub: 'predicts Q for (3,1) up', v: `Q = ${fmt(r.diagQ[i])}` }, { x: E.W - 8 - bw, y: 8, w: bw, h: band - 16, t: 'Target network', sub: 'builds the target', v: `y = ${fmt(r.diagY[i])}` }];
    for (const b of boxes) {
      rect(E, b.x, b.y, b.w, b.h, { fill: 'var(--bg-raised)', stroke: RULE, sw: 1.25, rx: 6 });
      text(E, b.t, b.x + 10, b.y + 20, { weight: 600, size: nar ? 12 : 13 });
      text(E, b.sub, b.x + 10, b.y + 38, { fill: MUTED, size: 11 });
      text(E, b.v, b.x + 10, b.y + b.h - 12, { weight: 700, size: nar ? 15 : 18 });
    }
    const ax0 = 8 + bw + 6, ax1 = E.W - 8 - bw - 6, cy = 8 + (band - 16) / 2 - 6, copied = f.k % s.copy === 0;
    arrow(E, ax0, cy, ax1, cy, { stroke: copied ? 'var(--accent)' : RULE, width: copied ? 3 : 1.5 });
    text(E, copied ? 'copied now' : `copy every ${s.copy}`, (ax0 + ax1) / 2, cy - 8, { anchor: 'middle', size: 11, fill: copied ? 'var(--accent)' : MUTED, weight: copied ? 700 : 400 });
    const dw = (ax1 - ax0) * (r.weightDist[i] / (s.dmax || 1));
    rect(E, ax0, cy + 12, ax1 - ax0, 6, { fill: 'none', stroke: RULE, sw: 1 });
    rect(E, ax0, cy + 12, dw, 6, { fill: 'var(--viz-gradient)' });
    text(E, 'weights apart', (ax0 + ax1) / 2, cy + 34, { anchor: 'middle', size: 11, fill: MUTED });
    const vals = [...Array.from(r.diagQ), ...Array.from(r.diagY), s.qstar], lo = Math.floor((Math.min(...vals) - 0.05) * 5) / 5, hi = Math.ceil((Math.max(...vals) + 0.05) * 5) / 5;
    const cb = { x: 46, y: band + 22, w: E.W - 62, h: E.H - band - 22 - (nar ? 48 : 52) };
    const { sx, sy } = axes(E, cb, { lo: 0, hi: s.n, ticks: [0, 50, 100, 150, 200, 250, 300].filter((v) => v <= s.n), label: 'Update' }, { lo, hi, label: 'Q or target for (3,1) up' });
    for (let c = s.copy; c <= s.n; c += s.copy) line(E, sx(c), cb.y, sx(c), cb.y + cb.h, { stroke: RULE, sw: 1, dash: '2 3' });
    line(E, cb.x, sy(s.qstar), cb.x + cb.w, sy(s.qstar), { stroke: MUTED, sw: 1.25, dash: '5 4' });
    text(E, `Q* ${fmt(s.qstar)}`, cb.x + cb.w - 4, sy(s.qstar) - 5, { anchor: 'end', size: 11, fill: MUTED });
    let dq = '', dy = '';
    for (let j = 0; j <= i; j++) {
      dq += `${j ? 'L' : 'M'}${sx(j + 1).toFixed(1)} ${sy(r.diagQ[j]).toFixed(1)}`;
      dy += j ? `H${sx(j + 1).toFixed(1)}V${sy(r.diagY[j]).toFixed(1)}` : `M${sx(1).toFixed(1)} ${sy(r.diagY[0]).toFixed(1)}`;
    }
    path(E, dq, { stroke: 'var(--viz-cat-1)', sw: 2 });
    path(E, dy, { stroke: 'var(--viz-cat-2)', sw: 2.2 });
    const px = sx(f.k);
    line(E, px, cb.y, px, cb.y + cb.h, { stroke: TEXT, sw: 1, op: 0.45 });
    poly(E, [[px, cb.y + cb.h + 1], [px - 4, cb.y + cb.h + 8], [px + 4, cb.y + cb.h + 8]], { fill: TEXT, op: 0.6 });
  },
  label: (s) => `Two network boxes, online and target, with the distance between their weights, above a chart of the online prediction and the target for one move over ${s.n} updates; the target changes only when the weights are copied every ${s.copy} updates.`,
};
export default mode;
