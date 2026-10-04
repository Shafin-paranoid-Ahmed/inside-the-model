/**
 * ppo-epochs (ppo step 5): one frozen batch of 64 samples, optimised for 4 epochs of shuffled mini-batches of 16.
 * Left: the batch (colour = sign of the frozen advantage, outline = the mini-batch being used). Right: histogram of
 * the 64 ratios (flat = zero surrogate gradient, still pushing = the rest), and the critic's loss per update.
 * Props: epsilon (0.2), epochs (4), minibatch (16), batch (64), lr (0.03), seed (31). Batch from PPO warmed up for 10 iterations.
 */
import { makeGrid } from '../../../lib/rl/gridworld';
import { batchStats, clippedSurrogate, driftRuns } from '../../../lib/rl/ppo';
import { int, key, num, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { ACCENT, BG, MUTED, TEXT, axes, circle, clamp, ease, fmt, lerp, mix, path, rect, signedFill, text, type Env } from '../_rl/svg';

const LEAD = 1.8, UPD = 0.75, BINS = 18, LO = 0.6, HI = 1.5;
interface St { ratios: Float64Array; act: Int32Array; flat: Int32Array; oob: number; flatFrac: number; oobActive: number; sur: number; kl: number }
interface Scene { n: number; adv: Float64Array; idx: Int32Array[]; st: St[]; vl: number[]; eps: number; ups: number; per: number; pos: number; hi: number }
interface F extends Frame { k: number; kh: number; w: number; n: number; mb: boolean }

function stateOf(ratios: ArrayLike<number>, adv: ArrayLike<number>, eps: number, sur: number, kl: number): St {
  const act = new Int32Array(BINS), flat = new Int32Array(BINS), n = ratios.length;
  let oob = 0, fl = 0, oa = 0;
  for (let i = 0; i < n; i++) {
    const r = ratios[i], c = clippedSurrogate(r, adv[i], eps), b = clamp(Math.floor(((r - LO) / (HI - LO)) * BINS), 0, BINS - 1), out = r < 1 - eps || r > 1 + eps;
    if (c.dRho === 0 && adv[i] !== 0) { flat[b]++; fl++; } else act[b]++;
    if (out) oob++;
    if (out && !(c.dRho === 0 && adv[i] !== 0)) oa++;
  }
  return { ratios: Float64Array.from(ratios), act, flat, oob: oob / n, flatFrac: fl / n, oobActive: oa, sur, kl };
}

const mode: ModeImpl<Scene, F> = {
  setup(o: Opts, kit) {
    const eps = num(o.epsilon, 0.2), epochs = int(o.epochs, 4), mini = int(o.minibatch, 16), lr = num(o.lr, 0.03), n = int(o.batch, 64), seed = int(o.seed, 31);
    const gamma = num(o.gamma, 0.9), sr = num(o.stepReward, -0.04), sl = num(o.slip, 0.2);
    const run = kit.memo(key('ppo-epochs', gamma, sr, sl, eps, epochs, mini, lr, n, seed), () => {
      const d = driftRuns(makeGrid({ gamma, stepReward: sr, slip: sl }), { warmIterations: 10, seed, lr, epsilons: [eps], epochs, minibatch: mini, batch: n });
      const it = d.runs[0].it, b = d.batch;
      const st = [stateOf(batchStats(b, d.theta0, eps).ratios, b.adv, eps, batchStats(b, d.theta0, eps).surrogate, 0), ...it.updates.map((u) => stateOf(u.stats.ratios, b.adv, eps, u.stats.surrogate, u.stats.kl))];
      return { adv: b.adv, idx: it.updates.map((u) => u.idx), st, vl: it.updates.map((u) => u.valueLoss), pos: Array.from(b.adv).filter((x) => x > 0).length };
    });
    return { n, ...run, eps, ups: run.idx.length, per: Math.max(1, run.idx.length / epochs), hi: Math.max(...run.vl) * 1.15 };
  },
  timeline(s) {
    const mk = [{ at: 0, label: 'Frozen batch' }];
    for (let e = 0; e * s.per < s.ups; e++) mk.push({ at: LEAD + e * s.per * UPD, label: `Epoch ${e + 1}` });
    return { duration: LEAD + s.ups * UPD + 2.5, markers: mk };
  },
  frameAt(t, s) {
    const u = clamp((t - LEAD) / UPD, 0, s.ups), ended = u >= s.ups, k = ended ? s.ups - 1 : Math.floor(u), fr = ended ? 1 : u - k;
    const live = t >= LEAD && !ended, mb = live && fr < 0.55, n = t < LEAD ? 0 : ended ? s.ups : k + (fr >= 0.55 ? 1 : 0);
    const kh = ended ? s.ups : t < LEAD ? 0 : k, w = live ? ease((fr - 0.4) / 0.3) : 0;
    const st = s.st[n], pct = (x: number) => `${Math.round(x * 100)}%`, ep = Math.floor(k / s.per) + 1, inEp = (k % s.per) + 1;
    let status: string;
    if (t < LEAD) status = `This batch of ${s.n} steps is frozen: its saved probabilities, advantages and critic targets stay fixed. ${s.pos} advantages are positive (green) and ${s.n - s.pos} negative (red). All ratios start at 1.`;
    else if (mb) status = `Epoch ${ep}, step ${inEp} of ${s.per}: PPO takes the ${s.idx[k].length} outlined samples and moves the policy one step up their clipped objective.`;
    else status = `${n === s.ups ? 'After all' : 'After'} ${n} update${n === 1 ? '' : 's'}: ${pct(st.oob)} of the ratios are outside the band, ${pct(st.flatFrac)} of the samples are flat, and ${st.oobActive} outside the band ${st.oobActive === 1 ? 'is' : 'are'} still pushing.`;
    const vl = n ? s.vl[n - 1] : NaN;
    return { k, kh, w, n, mb, status, readouts: [['Update', `${n} of ${s.ups}`], ['Mean objective', fmt(st.sur, 4)], ['Critic loss', n ? fmt(vl, 3) : '—'], ['Outside band', pct(st.oob)], ['Flat', pct(st.flatFrac)], ['Outside but pushing', String(st.oobActive)]] };
  },
  draw(E: Env, f, s) {
    const nw = E.narrow, W = E.W, H = E.H, side = 8;
    const gb = nw ? { x: 12, y: 30, w: W - 24, h: H * 0.22 } : { x: 14, y: 34, w: W * 0.3, h: W * 0.3 };
    const cs = Math.floor(Math.min(gb.w / side, gb.h / Math.ceil(s.n / side))), gx = nw ? gb.x + (gb.w - cs * side) / 2 : gb.x;
    text(E, 'The batch (one square per step)', gb.x, gb.y - 12, { size: 12, fill: MUTED });
    text(E, '● flat   ○ outside the band, still pushing', gb.x, gb.y + cs * Math.ceil(s.n / side) + 16, { size: 11, fill: MUTED });
    const k = Math.min(s.ups - 1, f.k), cur = s.st[f.n], sel = new Set<number>(f.mb ? Array.from(s.idx[k]) : []);
    const eps = s.eps;
    for (let i = 0; i < s.n; i++) {
      const x = gx + (i % side) * cs, y = gb.y + Math.floor(i / side) * cs;
      rect(E, x + 1, y + 1, cs - 2, cs - 2, { fill: signedFill(s.adv[i], 0.4), stroke: sel.has(i) ? ACCENT : BG, sw: sel.has(i) ? 2.5 : 1 });
      const r = cur.ratios[i], c = clippedSurrogate(r, s.adv[i], eps), flat = c.dRho === 0 && s.adv[i] !== 0;
      if (flat) circle(E, x + cs / 2, y + cs / 2, Math.max(2.5, cs * 0.14), { fill: TEXT });
      else if (r < 1 - eps || r > 1 + eps) circle(E, x + cs / 2, y + cs / 2, Math.max(3.5, cs * 0.2), { fill: 'none', stroke: TEXT, sw: 2 });
    }
    // histogram of ratios: still pushing (forward colour) and flat (grey), stacked
    const hb = nw ? { x: 40, y: gb.y + cs * Math.ceil(s.n / side) + 40, w: W - 56, h: H * 0.27 } : { x: gb.x + gb.w + 56, y: 40, w: W - gb.x - gb.w - 74, h: H * 0.44 };
    const mx = Math.max(8, ...s.st.map((x) => Math.max(...Array.from(x.act, (v, i) => v + x.flat[i]))));
    const { sx, sy } = axes(E, hb, { lo: LO, hi: HI, ticks: [0.6, 0.8, 1, 1.2, 1.4], label: 'Probability ratio ρ' }, { lo: 0, hi: mx, label: 'Number of samples' });
    rect(E, sx(1 - eps), hb.y, sx(1 + eps) - sx(1 - eps), hb.h, { fill: mix(ACCENT, BG, 16) });
    const from = s.st[f.kh], to = s.st[Math.min(s.ups, f.kh + 1)], bw = hb.w / BINS;
    for (let i = 0; i < BINS; i++) {
      const a2 = lerp(from.act[i], to.act[i], f.w), f2 = lerp(from.flat[i], to.flat[i], f.w);
      if (a2 > 0.01) rect(E, hb.x + i * bw + 1, sy(a2), bw - 2, sy(0) - sy(a2), { fill: 'var(--viz-forward)' });
      if (f2 > 0.01) rect(E, hb.x + i * bw + 1, sy(a2 + f2), bw - 2, sy(a2) - sy(a2 + f2), { fill: MUTED });
    }
    // critic loss per update
    const lb = nw ? { x: 40, y: hb.y + hb.h + 62, w: W - 56, h: Math.max(56, H - (hb.y + hb.h + 62) - 44) } : { x: hb.x, y: hb.y + hb.h + 62, w: hb.w, h: Math.max(56, H - (hb.y + hb.h + 62) - 44) };
    const { sx: lx, sy: ly } = axes(E, lb, { lo: 0, hi: s.ups, ticks: [0, s.ups / 4, s.ups / 2, (3 * s.ups) / 4, s.ups], label: 'Update' }, { lo: 0, hi: s.hi, label: 'Critic loss on the mini-batch' });
    let d = '';
    for (let j = 0; j < f.n; j++) { d += `${d ? 'L' : 'M'}${lx(j + 1).toFixed(1)} ${ly(s.vl[j]).toFixed(1)}`; }
    if (d) path(E, d, { stroke: 'var(--viz-gradient)', sw: 2 });
    if (f.n) circle(E, lx(f.n), ly(s.vl[f.n - 1]), 4, { fill: ACCENT, stroke: BG, sw: 2 });
  },
  label: (s) => `A frozen batch of ${s.n} samples, a histogram of their probability ratios over ${s.ups} mini-batch updates, and the critic loss per update.`,
};
export default mode;
