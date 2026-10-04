/**
 * dqn-update (deep-rl step 2): one real transition (the agent at (2,3) asks for right) becomes a target
 * y = r + 0.9 max Q(s', .), the squared error is backpropagated once with y held constant, and the grid's Q-values
 * are redrawn from the new weights. SGD, no replay, no target network.
 * Props: hidden (16), seed (17), lr (0.05), focus ("2,3"), action ("right").
 */
import { createRng } from '../../../lib/math/rng';
import { ACTIONS, cellName, idx, step, xy } from '../../../lib/rl/gridworld';
import { allQ, initNet, tdLossGrad, tdTarget, type Transition } from '../../../lib/rl/dqn';
import { int, num, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { ACCENT, MUTED, RULE, TEXT, arrow, circle, ease, fmt, line, lerp, poly, rect, seg, text, type Env } from '../_rl/svg';
import { cellXY, drawAgent, drawGrid, drawQ, gridIn } from './draw';
import { GAMMA, dqnGrid } from './dqn-common';
import { actionOf, focusOf } from './sim';

const T = [0, 1.8, 4.0, 5.8, 8.0], END = 11.5;
const NAMES = ['Move', 'Target', 'Error', 'Update', 'After'];
interface Scene {
  t: Transition; Qb: Float64Array; Qa: Float64Array; qsa: number; qsa2: number; nextBest: number; y: number; loss: number; dw: number; lr: number;
  nb: { s: number; before: number; after: number }; states: number[];
}
interface F extends Frame { p: number; alpha: number; fx: number; fy: number; time: number }

const mode: ModeImpl<Scene, F> = {
  setup(o: Opts) {
    const g = dqnGrid(o), H = int(o.hidden, 16), w0 = initNet(H, int(o.seed, 17)), lr = num(o.lr, 0.05), s = focusOf(o, idx(2, 3)), a = actionOf(o, 1);
    // draw outcomes until the move is not a slip, so the transition is "move right from (2,3)" as asked
    let tr: Transition = { s, a, r: 0, s2: s, done: false };
    for (let k = 0; k < 50; k++) { const r = step(g, s, a, createRng(int(o.seed, 17) + k)); tr = { s, a, r: r.r, s2: r.s2, done: r.done }; if (!r.slipped) break; }
    const grad = new Float64Array(w0.length), res = tdLossGrad(w0, w0, H, [tr], GAMMA, grad), w1 = w0.slice();
    let dw = 0;
    for (let i = 0; i < w1.length; i++) { w1[i] -= lr * grad[i]; dw = Math.max(dw, Math.abs(lr * grad[i])); }
    const Qb = allQ(g, w0, H), Qa = allQ(g, w1, H), y = tdTarget(w0, H, tr, GAMMA);
    let nbS = g.states.find((x) => x !== tr.s) ?? 0, best = -1;
    for (const x of g.states) if (x !== tr.s && Math.abs(Qa[x * 4 + a] - Qb[x * 4 + a]) > best) { best = Math.abs(Qa[x * 4 + a] - Qb[x * 4 + a]); nbS = x; }
    return {
      t: tr, Qb, Qa, qsa: Qb[s * 4 + a], qsa2: Qa[s * 4 + a], y, nextBest: tr.done ? 0 : (y - tr.r) / GAMMA, loss: res.loss, dw, lr,
      nb: { s: nbS, before: Qb[nbS * 4 + a], after: Qa[nbS * 4 + a] }, states: g.states,
    };
  },
  timeline: () => ({ duration: END, markers: T.map((at, i) => ({ at, label: NAMES[i] })) }),
  frameAt(t, s) {
    const p = T.filter((x) => t >= x).length - 1, { t: tr } = s, [x0, y0] = xy(tr.s), [x1, y1] = xy(tr.s2), m = ease(t / 1.4);
    const alpha = ease((t - T[3]) / (T[4] - T[3])), a = ACTIONS[tr.a], S = cellName(tr.s), S2 = cellName(tr.s2), err = s.qsa - s.y;
    const status = [
      `The agent is at ${S}, asks for ${a} and ends up at ${S2}. The move costs ${fmt(tr.r)}.`,
      tr.done ? `The move ended the episode, so the target is just the reward, y = ${fmt(s.y)}.` : `The move says what Q(${S}, ${a}) should be: the reward ${fmt(tr.r)} plus ${GAMMA} times the best Q-value the network gives ${S2}, ${fmt(s.nextBest)}. That target is y = ${fmt(s.y)}.`,
      `The network currently predicts ${fmt(s.qsa)} for ${a} at ${S}, so it is off by ${fmt(err)}. The loss is half that error squared, ${fmt(s.loss, 3)}.`,
      `Backpropagation turns the error into a small push on every weight (learning rate ${s.lr}), treating the target ${fmt(s.y)} as a fixed number. The largest push on one weight is ${fmt(s.dw, 3)}.`,
      `After the update the prediction is ${fmt(s.qsa2)}. The same weights also moved ${a} at ${cellName(s.nb.s)}, a cell the agent did not visit, from ${fmt(s.nb.before)} to ${fmt(s.nb.after)}.`,
    ][p];
    return { p, alpha, time: t, fx: lerp(x0, x1, m), fy: lerp(y0, y1, m), status, readouts: [['Step', NAMES[p]], ['Prediction', fmt(p >= 4 ? s.qsa2 : s.qsa)], ['Target y', p >= 1 ? fmt(s.y) : '…']] };
  },
  draw(E: Env, f, s) {
    const nar = E.narrow, gb = nar ? { x: 8, y: 22, w: E.W - 16, h: E.H * 0.34 } : { x: 12, y: 26, w: E.W * 0.46, h: E.H - 36 };
    const pb = nar ? { x: 12, y: gb.y + gb.h + 16, w: E.W - 24, h: E.H - gb.y - gb.h - 24 } : { x: gb.x + gb.w + 26, y: 8, w: E.W - gb.x - gb.w - 40, h: E.H - 16 };
    const G = gridIn(gb), tr = s.t;
    text(E, 'Q-values of every cell', gb.x, 14, { fill: MUTED });
    drawGrid(E, G);
    drawQ(E, G, s.Qb, { vmax: 1.3, best: false, numbers: false });
    if (f.alpha > 0) drawQ(E, G, s.Qa, { vmax: 1.3, best: false, numbers: false, show: () => f.alpha });
    const c = cellXY(G, tr.s), a = tr.a, tri = [[0, -1], [1, 0], [0, 1], [-1, 0]][a];
    rect(E, c.x + 1.5, c.y + 1.5, G.cs - 3, G.cs - 3, { fill: 'none', stroke: ACCENT, sw: 2.5 });
    arrow(E, c.cx, c.cy, c.cx + tri[0] * G.cs * 0.38, c.cy + tri[1] * G.cs * 0.38, { stroke: TEXT, width: 2 });
    const n = cellXY(G, s.nb.s);
    if (f.p >= 4) rect(E, n.x + 3, n.y + 3, G.cs - 6, G.cs - 6, { fill: 'none', stroke: TEXT, sw: 1.5, dash: '4 3' });
    drawAgent(E, G, f.fx, f.fy);
    // number line: where the prediction is, where the target is
    const ax = { x: pb.x + 40, w: pb.w - 80, y: pb.y + 46 }, X = (v: number) => ax.x + ((Math.max(-1.2, Math.min(1.2, v)) + 1.2) / 2.4) * ax.w;
    text(E, 'Prediction and target for this move', pb.x, pb.y + 10, { fill: MUTED });
    line(E, ax.x, ax.y, ax.x + ax.w, ax.y, { stroke: RULE, sw: 1.5 });
    for (const v of [-1, 0, 1]) { line(E, X(v), ax.y - 4, X(v), ax.y + 4, { stroke: RULE }); text(E, fmt(v, 0), X(v), ax.y + 30, { anchor: 'middle', fill: MUTED, size: 11 }); }
    const fade = (start: number) => ease((f.time - start) / 0.5);
    if (f.time >= 4.0 && f.time >= 2.8) arrow(E, X(s.qsa), ax.y, X(s.y), ax.y, { stroke: 'var(--viz-gradient)', width: 2.5, op: fade(4.0) });
    if (f.time >= 2.8) { poly(E, [[X(s.y), ax.y - 3], [X(s.y) - 6, ax.y - 12], [X(s.y) + 6, ax.y - 12]], { fill: TEXT, op: fade(2.8) }); text(E, 'target y', X(s.y), ax.y - 17, { anchor: 'middle', size: 11, op: fade(2.8) }); }
    if (f.time >= 4.0) { circle(E, X(s.qsa), ax.y, 6, { fill: ACCENT, op: fade(4.0) }); text(E, 'now', X(s.qsa), ax.y + 18, { anchor: 'middle', size: 11, op: fade(4.0) }); }
    if (f.time >= T[4]) { circle(E, X(s.qsa2), ax.y, 6, { fill: 'none', stroke: ACCENT, sw: 2.5, op: fade(T[4]) }); text(E, 'after', X(s.qsa2), ax.y + 18, { anchor: 'middle', size: 11, op: fade(T[4]) }); }
    const rows: [string, string, number][] = [
      ['Reward r', fmt(tr.r), 1.0], [`Best next Q, max Q(${cellName(tr.s2)}, ·)`, fmt(s.nextBest), 1.8], [`Target y = r + ${GAMMA} × best`, fmt(s.y), 2.8],
      [`Prediction Q(${cellName(tr.s)}, ${ACTIONS[a]})`, fmt(s.qsa), 4.0], ['Error (prediction − y)', fmt(s.qsa - s.y), 4.0], ['Loss, half the error squared', fmt(s.loss, 3), 4.8],
      ['Largest weight change', fmt(s.dw, 3), 5.8], ['Prediction after the update', fmt(s.qsa2), T[4]], [`${ACTIONS[a]} at ${cellName(s.nb.s)}, before → after`, `${fmt(s.nb.before)} → ${fmt(s.nb.after)}`, T[4]],
    ];
    const y0 = ax.y + 52, rh = nar ? 19 : 22;
    rows.forEach(([k, v, st], i) => {
      if (f.time < st) return;
      const op = fade(st), yy = y0 + i * rh;
      text(E, k, pb.x, yy, { size: 12, op, fill: i === 7 || i === 8 ? TEXT : MUTED }); text(E, v, pb.x + pb.w, yy, { size: 12, anchor: 'end', weight: 600, op });
    });
  },
  label: (s) => `A 4 by 3 grid coloured by a network's Q-values and a list of the numbers in one update: the move from ${cellName(s.t.s)} to ${cellName(s.t.s2)}, its target, the error and the prediction after one gradient step.`,
};
export default mode;
