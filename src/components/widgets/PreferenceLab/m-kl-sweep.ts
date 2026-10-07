/**
 * kl-sweep (rlhf step 6): four PPO runs against the same reward model that differ only in the KL penalty beta, advancing
 * together. The plane is KL to the SFT model (square-root scale, so 0.29 and 13 both fit) against gold reward; each run
 * leaves a path and a dot at its current state. The run with the highest gold reward so far is ringed.
 * Props: betas ("0.5,0.1,0.02,0"), iters (200), labels ("human"), seed (6 = PPO seed).
 * Measured (plan §7), final KL / proxy / gold: beta 0.5 0.290 / 3.971 / 2.762; 0.1 0.802 / 4.076 / 2.808 (best gold);
 *   0.02 13.01 / 4.719 / -0.440; 0 13.01 / 4.719 / -0.439.
 */
import { int, nums, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { ACCENT, BG, MUTED, TEXT, axes, circle, clamp, fmt, path, text, type Env } from '../_rl/svg';
import type { AlignPPOResult } from '../../../lib/align/ppo';
import { runsFor, quarterMarkers } from './rlhf-common';

const LEAD = 1.2, PACE = 0.08, HOLD = 2.5;
interface Scene { betas: number[]; runs: AlignPPOResult[]; iters: number; sq: Float64Array[]; xHi: number }
interface F extends Frame { n: number; best: number }
const bl = (b: number) => (b === 0 ? 'β = 0' : `β = ${b}`);

const mode: ModeImpl<Scene, F> = {
  async setup(o: Opts, kit) {
    const betas = nums(o.betas, [0.5, 0.1, 0.02, 0]), iters = int(o.iters, 200), runs = await runsFor(o, kit, betas, iters);
    const sq = runs.map((r) => Float64Array.from(r.kl, Math.sqrt));
    return { betas, runs, iters, sq, xHi: Math.max(...sq.map((a) => Math.max(...a))) * 1.08 };
  },
  timeline: (s) => ({ duration: LEAD + s.iters * PACE + HOLD, markers: [{ at: 0, label: 'Start' }, ...quarterMarkers(LEAD, PACE, s.iters)] }),
  frameAt(t, s) {
    const n = clamp(Math.round((t - LEAD) / PACE), 0, s.iters);
    let best = 0;
    s.runs.forEach((r, j) => { if (r.gold[n] > s.runs[best].gold[n] + 1e-12) best = j; });
    const status = n === 0
      ? `All four runs start from the SFT model, so KL is 0 and gold is ${fmt(s.runs[0].gold[0], 2)}. They differ only in the penalty β.`
      : `After ${n} update${n === 1 ? '' : 's'}, gold reward: ${s.betas.map((b, j) => `${bl(b)} ${fmt(s.runs[j].gold[n], 2)}`).join(', ')}. The highest so far is ${bl(s.betas[best])}.`;
    return { n, best, status, readouts: s.betas.map((b, j): [string, string] => [bl(b), `KL ${fmt(s.runs[j].kl[n], 2)}, proxy ${fmt(s.runs[j].proxy[n], 2)}, gold ${fmt(s.runs[j].gold[n], 2)}`]) };
  },
  draw(E: Env, f, s) {
    const nw = E.narrow, b = { x: nw ? 40 : 56, y: 28, w: E.W - (nw ? 40 : 56) - (nw ? 14 : 20), h: E.H - 28 - 50 }, n = f.n;
    const root = (v: number) => v * v;
    const { sx, sy } = axes(E, b, { lo: 0, hi: s.xHi, ticks: [0, 0.5, 1, 2, 3].filter((v) => v < s.xHi), fmt: (v) => fmt(root(v), root(v) < 1 && v ? 2 : 0), label: 'KL to the SFT model (square-root scale)' }, { lo: -1, hi: 3.5, ticks: [-1, 0, 1, 2, 3], label: 'Gold reward' });
    s.runs.forEach((r, j) => {
      let d = '';
      for (let i = 0; i <= n; i++) d += `${d ? 'L' : 'M'}${sx(s.sq[j][i]).toFixed(1)} ${clamp(sy(r.gold[i]), b.y, b.y + b.h).toFixed(1)}`;
      path(E, d, { stroke: MUTED, sw: 2, op: 0.7 });
    });
    // labels keep at least 14 px apart (the beta 0.02 and beta 0 runs end on the same point)
    const dots = s.runs.map((r, j) => ({ j, x: sx(s.sq[j][n]), y: clamp(sy(r.gold[n]), b.y, b.y + b.h) })).sort((a, c) => a.y - c.y);
    let last = -99;
    for (const d of dots) {
      circle(E, d.x, d.y, 5, { fill: TEXT, stroke: d.j === f.best ? ACCENT : BG, sw: d.j === f.best ? 3 : 2 });
      const ly = Math.max(d.y + 4, last + 14);
      last = ly;
      const flip = d.x > b.x + b.w - 70;
      text(E, bl(s.betas[d.j]), d.x + (flip ? -10 : 10), ly, { size: 12, weight: 650, halo: true, anchor: flip ? 'end' : 'start' });
    }
  },
  label: (s) => `Gold reward against KL to the SFT model for PPO runs with penalties ${s.betas.join(', ')}. After ${s.iters} updates gold ends at ${s.runs.map((r) => fmt(r.gold[s.iters], 2)).join(', ')} respectively.`,
};
export default mode;
