/**
 * ppo-advantage (ppo step 1): one episode of a collected batch. Each step shows r, the frozen critic's V(s) and V(s'),
 * and the TD error delta = r + gamma V(s') - V(s). Then the advantage of step 1 is built as the discounted sum of
 * later errors, (gamma lambda)^l delta, and compared with lambda = 0 (delta alone).
 * Props: seed (29), gamma (0.9), lambda (0.95), batch (64). The batch comes from a policy PPO trained for 5 iterations.
 */
import { ACTIONS, ARROWS, cellName } from '../../../lib/rl/gridworld';
import { type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { ACCENT, MUTED, RULE, clamp, fmt, inkOn, line, rect, signFmt, signedFill, text, type Env } from '../_rl/svg';
import { warmBatch } from './ppo-batch';

const LEAD = 0.5, ROW = 0.9, BAR = 0.8, HOLD = 3.5;
interface Scene { T: number; gamma: number; lambda: number; s: number[]; a: number[]; r: number[]; v: number[]; v2: number[]; d: number[]; A: number[]; w: number[]; goal: boolean }
interface F extends Frame { rows: number; bars: number; sum: number; cmp: boolean }

const mode: ModeImpl<Scene, F> = {
  setup(o: Opts, kit) {
    const { g, b, lambda } = warmBatch(o, kit, 5, 29);
    let T = 0;
    while (T < b.s.length - 1 && !b.terminated[T] && !b.truncated[T]) T++;
    T++;
    const cut = (x: ArrayLike<number>) => Array.from({ length: T }, (_, i) => x[i]);
    const d = cut(b.delta), c = g.gamma * lambda;
    return { T, gamma: g.gamma, lambda, s: cut(b.s), a: cut(b.a), r: cut(b.r), v: cut(b.v), v2: cut(b.v2), d, A: cut(b.adv), w: d.map((x, l) => c ** l * x), goal: !!b.terminated[T - 1] };
  },
  timeline(s) {
    const p1 = LEAD + s.T * ROW, p2 = p1 + s.T * BAR;
    return { duration: p2 + HOLD, markers: [{ at: 0, label: 'Error of each step' }, { at: p1, label: 'Add up later errors' }, { at: p2, label: 'One step or many' }] };
  },
  frameAt(t, s) {
    const p1 = LEAD + s.T * ROW, p2 = p1 + s.T * BAR;
    const rows = t < LEAD ? 0 : Math.min(s.T, Math.floor((t - LEAD) / ROW) + 1);
    const bars = t < p1 ? 0 : Math.min(s.T, Math.floor((t - p1) / BAR) + 1), cmp = t >= p2;
    let sum = 0;
    for (let l = 0; l < bars; l++) sum += s.w[l];
    let status: string;
    if (rows === 0) status = `A batch of steps was collected. Here is one episode, ${s.T} steps long. The critic has already estimated a value V for every cell it visits.`;
    else if (!bars) {
      const k = rows - 1, ter = k === s.T - 1 && s.goal;
      status = `Step ${rows}: at ${cellName(s.s[k])} the agent went ${ACTIONS[s.a[k]]} and earned ${fmt(s.r[k])}. The critic expected ${fmt(s.v[k], 3)} from here and values the next cell at ${ter ? '0 (the episode ended)' : fmt(s.v2[k], 3)}, so the error is ${fmt(s.r[k])} + ${fmt(s.gamma, 1)} × ${fmt(ter ? 0 : s.v2[k], 3)} − (${fmt(s.v[k], 3)}) = ${signFmt(s.d[k], 3)}.`;
    } else if (!cmp) {
      const l = bars - 1;
      status = `Step ${l + 1}'s error ${signFmt(s.d[l], 3)} is counted for step 1 with weight (${fmt(s.gamma, 1)} × ${fmt(s.lambda, 2)})^${l} = ${fmt((s.gamma * s.lambda) ** l, 3)}, so it adds ${signFmt(s.w[l], 3)}. Step 1's advantage so far: ${signFmt(sum, 3)}.`;
    } else status = `With λ = 0 the advantage of step 1 is its own error, ${signFmt(s.d[0], 3)}. With λ = ${fmt(s.lambda, 2)} it also collects the later errors and reaches ${signFmt(s.A[0], 3)}.`;
    const readouts: [string, string][] = rows === 0 ? [] : [['Step 1 error δ', signFmt(s.d[0], 3)], ['Step 1 advantage', bars ? signFmt(cmp ? s.A[0] : sum, 3) : '—'], ['Episode', `${s.T} steps`]];
    return { rows, bars, sum, cmp, status, readouts };
  },
  draw(E: Env, f, s) {
    const nw = E.narrow, T = s.T, rh = nw ? 20 : 24, size = nw ? 11 : 12;
    const x0 = nw ? 8 : 14, y0 = 24, head = ['#', 'Cell', 'Move', 'r', 'V(s)', 'V(s′)', 'δ', 'A'];
    const px = nw ? [0, 16, 58, 124, 162, 206, 258, 306] : [0, 26, 78, 164, 214, 272, 338, 400];
    const an: ('start' | 'end')[] = ['start', 'start', 'start', 'end', 'end', 'end', 'end', 'end'];
    const cx = (j: number) => x0 + px[j];
    head.forEach((h, j) => text(E, h, cx(j), y0, { size, fill: MUTED, anchor: an[j], weight: j >= 6 ? 650 : 400 }));
    line(E, x0, y0 + 7, cx(7) + 4, y0 + 7, { stroke: RULE, sw: 1 });
    const vmax = 0.9, bw = nw ? 46 : 52;
    const cell = (j: number, y: number, label: string, v?: number) => {
      if (v !== undefined) rect(E, cx(j) - bw + 4, y - rh * 0.7, bw, rh - 3, { fill: signedFill(v, vmax) });
      text(E, label, cx(j), y, { size, anchor: 'end', mono: true, fill: v !== undefined ? inkOn(Math.min(1, Math.abs(v) / vmax)) : undefined });
    };
    for (let k = 0; k < T; k++) {
      const y = y0 + 8 + (k + 1) * rh - 6;
      if (k === 0 && f.rows > 0) rect(E, x0 - 4, y - rh * 0.78, cx(7) - x0 + 12, rh, { fill: 'none', stroke: ACCENT, sw: 1.5 });
      if (k >= f.rows) continue;
      text(E, String(k + 1), cx(0), y, { size, fill: MUTED });
      text(E, cellName(s.s[k]), cx(1), y, { size, mono: true });
      text(E, nw ? ARROWS[s.a[k]] : `${ARROWS[s.a[k]]} ${ACTIONS[s.a[k]]}`, cx(2), y, { size });
      cell(3, y, fmt(s.r[k]));
      cell(4, y, fmt(s.v[k], 3));
      cell(5, y, k === T - 1 && s.goal ? '0' : fmt(s.v2[k], 3));
      cell(6, y, signFmt(s.d[k], 3), s.d[k]);
      if (f.cmp) cell(7, y, signFmt(s.A[k], 3), s.A[k]);
    }
    // lower (phone) or right panel: how much each step's error adds to step 1's advantage
    const by = y0 + 8 + (T + 1) * rh + 26;
    const pb = nw ? { x: 14, y: by, w: E.W - 28, h: E.H - by - 16 } : { x: cx(7) + 40, y: 44, w: E.W - cx(7) - 54, h: E.H - 70 };
    text(E, 'Adds to step 1’s advantage', pb.x, pb.y - 14, { size, fill: MUTED });
    const bh = Math.min(26, (pb.h - 14) / T), zero = pb.x + pb.w * 0.3, scale = (pb.w * 0.6) / 0.9;
    line(E, zero, pb.y - 4, zero, pb.y + T * bh, { stroke: RULE, sw: 1 });
    for (let l = 0; l < Math.min(T, f.bars); l++) {
      const y = pb.y + l * bh, wv = s.w[l];
      rect(E, wv >= 0 ? zero : zero + wv * scale, y + 3, Math.max(1.5, Math.abs(wv) * scale), bh - 6, { fill: 'var(--viz-gradient)' });
      text(E, `step ${l + 1}`, pb.x, y + bh * 0.62, { size: 11, fill: MUTED });
      text(E, signFmt(wv, 3), wv >= 0 ? zero + Math.abs(wv) * scale + 6 : zero + 6, y + bh * 0.62, { size: 11, mono: true });
    }
    if (f.bars) {
      const y = pb.y + T * bh + 8, ex = zero + clamp(f.sum, 0, 0.9) * scale;
      line(E, zero, y, ex, y, { stroke: ACCENT, sw: 3 });
      text(E, `total ${signFmt(f.cmp ? s.A[0] : f.sum, 3)}`, ex + 6, y + 4, { size: 11, mono: true, fill: ACCENT });
    }
  },
  label: (s) => `A table of one ${s.T}-step episode with reward, critic values and the error of each step, beside bars showing how each step's error adds to the advantage of step 1 (${signFmt(s.d[0], 3)} with lambda 0, ${signFmt(s.A[0], 3)} with lambda ${s.lambda}).`,
};
export default mode;
