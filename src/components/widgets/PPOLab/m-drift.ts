/**
 * drift (ppo step 6): the SAME frozen batch and starting policy optimised with three clip widths. One strip per epsilon
 * shows all 64 probability ratios (dot colour = sign of the sample's advantage; the band is shaded); the line chart
 * shows KL(old || new) averaged over the batch's states after every mini-batch update.
 * Props: epsilons ("0.1,0.2,0.4"), epochs (4), minibatch (16), batch (64), lr (0.03), seed (31), ratios ("0.6,1.6").
 * The batch comes from a policy PPO trained for 10 iterations (a uniform-policy batch has only negative advantages).
 */
import { makeGrid } from '../../../lib/rl/gridworld';
import { batchStats, driftRuns } from '../../../lib/rl/ppo';
import { int, key, num, nums, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { ACCENT, BG, MUTED, axes, circle, clamp, ease, fmt, lerp, line, mix, path, rect, text, type Env } from '../_rl/svg';

const LEAD = 1.8, UPD = 0.75, CAT = ['--viz-cat-1', '--viz-cat-2', '--viz-cat-3'];
interface Run { eps: number; ratios: Float64Array[]; min: number[]; max: number[]; kl: number[]; oob: number[] }
interface Scene { adv: Float64Array; runs: Run[]; ups: number; per: number; lo: number; hi: number; klMax: number }
interface F extends Frame { kh: number; w: number; n: number }

const mode: ModeImpl<Scene, F> = {
  setup(o: Opts, kit) {
    const epochs = int(o.epochs, 4), mini = int(o.minibatch, 16), lr = num(o.lr, 0.03), n = int(o.batch, 64), seed = int(o.seed, 31), epsilons = nums(o.epsilons, [0.1, 0.2, 0.4]);
    const gamma = num(o.gamma, 0.9), [lo, hi] = nums(o.ratios, [0.6, 1.6]);
    const d = kit.memo(key('ppo-drift', gamma, epsilons, epochs, mini, lr, n, seed), () => {
      const g = makeGrid({ gamma }), r = driftRuns(g, { warmIterations: 10, seed, lr, epsilons, epochs, minibatch: mini, batch: n });
      const runs: Run[] = r.runs.map(({ eps, it }) => {
        const all = [batchStats(r.batch, r.theta0, eps), ...it.updates.map((u) => u.stats)];
        return { eps, ratios: all.map((x) => x.ratios), min: all.map((x) => x.min), max: all.map((x) => x.max), kl: all.map((x) => x.kl), oob: all.map((x) => x.outOfBand) };
      });
      return { adv: r.batch.adv, runs };
    });
    const ups = d.runs[0].kl.length - 1;
    return { ...d, ups, per: Math.max(1, ups / epochs), lo, hi, klMax: Math.max(...d.runs.flatMap((r) => r.kl)) * 1.08 };
  },
  timeline(s) {
    const mk = [{ at: 0, label: 'Same batch, same start' }];
    for (let e = 0; e * s.per < s.ups; e++) mk.push({ at: LEAD + e * s.per * UPD, label: `Epoch ${e + 1}` });
    return { duration: LEAD + s.ups * UPD + 2.5, markers: mk };
  },
  frameAt(t, s) {
    const u = clamp((t - LEAD) / UPD, 0, s.ups), ended = u >= s.ups, k = ended ? s.ups - 1 : Math.floor(u), fr = ended ? 1 : u - k, live = t >= LEAD && !ended;
    const n = t < LEAD ? 0 : ended ? s.ups : k + (fr >= 0.55 ? 1 : 0), kh = ended ? s.ups : t < LEAD ? 0 : k, w = live ? ease((fr - 0.4) / 0.3) : 0;
    const names = s.runs.map((r) => fmt(r.eps, 1)).join(', ');
    const rows: [string, string][] = s.runs.map((r) => [`ε = ${fmt(r.eps, 2)}`, `ratios ${fmt(r.min[n], 2)} to ${fmt(r.max[n], 2)}, KL ${fmt(r.kl[n], 4)}`]);
    const status = n === 0
      ? `Three copies of the same policy start from the same frozen batch. They differ only in the clip width ε = ${names}. Every ratio is 1.`
      : `After ${n} update${n === 1 ? '' : 's'}: ` + s.runs.map((r) => `with ε = ${fmt(r.eps, 1)} the ratios span ${fmt(r.min[n], 2)} to ${fmt(r.max[n], 2)} (band ${fmt(1 - r.eps, 2)} to ${fmt(1 + r.eps, 2)})`).join('; ') + '.';
    return { kh, w, n, status, readouts: rows };
  },
  draw(E: Env, f, s) {
    const nw = E.narrow, W = E.W, H = E.H, R = s.runs.length, sh = nw ? 56 : 62;
    const sb = { x: nw ? 14 : 78, y: 40, w: W - (nw ? 28 : 78) - 16, h: sh * R };
    const sx = (r: number) => sb.x + ((clamp(r, s.lo, s.hi) - s.lo) / (s.hi - s.lo)) * sb.w;
    text(E, 'Probability ratio of each of the 64 samples', sb.x, sb.y - 14, { size: 12, fill: MUTED });
    s.runs.forEach((run, j) => {
      const y0 = sb.y + j * sh, cy = y0 + sh / 2;
      rect(E, sx(1 - run.eps), y0 + 2, sx(1 + run.eps) - sx(1 - run.eps), sh - 4, { fill: mix(ACCENT, BG, 16) });
      line(E, sx(1), y0 + 2, sx(1), y0 + sh - 2, { stroke: MUTED, sw: 1, dash: '3 3' });
      line(E, sb.x, y0 + sh, sb.x + sb.w, y0 + sh, { stroke: 'var(--rule)', sw: 1 });
      const a0 = run.ratios[f.kh], a1 = run.ratios[Math.min(s.ups, f.kh + 1)];
      for (let i = 0; i < s.adv.length; i++) {
        const jy = ((i * 0.6180339) % 1 - 0.5) * (sh - 16);
        circle(E, sx(lerp(a0[i], a1[i], f.w)), cy + jy, 3.2, { fill: s.adv[i] >= 0 ? 'var(--viz-pos)' : 'var(--viz-neg)', stroke: BG, sw: 0.75, op: 0.85 });
      }
      text(E, `ε ${fmt(run.eps, 1)}`, nw ? sb.x + 2 : 14, nw ? y0 + 14 : cy + 4, { size: 12, weight: 650, fill: `var(${CAT[j % 3]})` });
    });
    for (const v of [0.6, 0.8, 1, 1.2, 1.4, 1.6].filter((v) => v >= s.lo && v <= s.hi)) text(E, fmt(v, 1), sx(v), sb.y + sh * R + 14, { size: 11, anchor: 'middle', fill: MUTED });
    // KL chart
    const ky = sb.y + sh * R + (nw ? 52 : 56), kb = { x: nw ? 46 : 78, y: ky, w: W - (nw ? 46 : 78) - 16, h: Math.max(60, H - ky - 44) };
    const { sx: kx, sy: ky2 } = axes(E, kb, { lo: 0, hi: s.ups, ticks: [0, s.ups / 4, s.ups / 2, (3 * s.ups) / 4, s.ups], label: 'Update' }, { lo: 0, hi: s.klMax, label: 'KL(old ‖ new), mean over the batch' });
    s.runs.forEach((run, j) => {
      let d = '';
      for (let i = 0; i <= Math.min(s.ups, f.n); i++) d += `${d ? 'L' : 'M'}${kx(i).toFixed(1)} ${ky2(run.kl[i]).toFixed(1)}`;
      if (d) path(E, d, { stroke: `var(${CAT[j % 3]})`, sw: 2.25 });
      const i = Math.min(s.ups, f.n);
      circle(E, kx(i), ky2(run.kl[i]), 4, { fill: `var(${CAT[j % 3]})`, stroke: BG, sw: 2 });
      text(E, `ε ${fmt(run.eps, 1)}`, kx(s.ups) - 4, ky2(run.kl[s.ups]) - 6, { size: 11, anchor: 'end', fill: `var(${CAT[j % 3]})`, weight: 650 });
    });
  },
  label: (s) => `Probability ratios of a frozen batch after each update for clip widths ${s.runs.map((r) => r.eps).join(', ')}, and the KL divergence from the old policy over ${s.ups} updates.`,
};
export default mode;
