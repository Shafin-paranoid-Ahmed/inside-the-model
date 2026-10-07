/**
 * length (rlvr step 5): the strict-verifier run of step 4 (same props and seed), drawn as the mean response length in
 * words per digit count, with a sampled answer to a 4-digit problem (refreshed every 10 iterations). Props: iters (150),
 * G (8), problems (8), seed (5). Measured (seed 5): mean words at 0 and 150: 5.7 -> 10.0, 7.3 -> 18.8, 8.1 -> 27.7, 8.4 -> 37.2.
 */
import type { GRPORun } from '../../../lib/align/verifier';
import { type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { MUTED, axes, fmt, lerp, path, text, type Env } from '../_rl/svg';
import { candColor } from '../_align/draw';
import { iterAt, liveSample, runProps, runTimeline, sampleCard, trainRun } from './vr-common';

interface F extends Frame { it: number; k: number }
const D = [0, 1, 2, 3];

const mode: ModeImpl<GRPORun, F> = {
  setup: (o: Opts, kit) => trainRun(runProps(o, 'strict'), kit),
  timeline: (r) => runTimeline(r.iters),
  frameAt(t, r) {
    const it = iterAt(t, r.iters), k = Math.floor(it + 1e-9), w = D.map((d) => fmt(r.words[k * 4 + d], 1)).join(' / ');
    return { it, k, status: `Iteration ${k}: the mean response is ${w} words long on 1, 2, 3 and 4 digit problems.`, readouts: [['Iteration', `${k} of ${r.iters}`], ['Mean words', w], ['Lines, 4-digit', fmt(r.lines[k * 4 + 3], 2)]] };
  },
  draw(E: Env, f, r) {
    const nw = E.narrow, W = E.W, H = E.H, n = r.iters, pm = nw ? 8 : 12;
    const cb = nw ? { x: 40, y: 22, w: W - 52, h: 200 } : { x: 46, y: 22, w: W * 0.56 - 56, h: H - 22 - 50 };
    const { sx, sy } = axes(E, cb, { lo: 0, hi: n, ticks: [0, n / 3, (2 * n) / 3, n], label: 'Training iteration' }, { lo: 0, hi: 40, ticks: [0, 10, 20, 30, 40], label: 'Mean response length (words)' });
    D.forEach((d) => {
      const v = (i: number) => r.words[i * 4 + d], k = Math.floor(f.it);
      let p = '';
      for (let i = 0; i <= k; i++) p += `${p ? 'L' : 'M'}${sx(i).toFixed(1)} ${sy(v(i)).toFixed(1)}`;
      if (f.it > k) p += `L${sx(f.it).toFixed(1)} ${sy(lerp(v(k), v(k + 1), f.it - k)).toFixed(1)}`;
      if (p) path(E, p, { stroke: candColor(d), sw: 2.25 });
      text(E, `${d + 1} digit${d ? 's' : ''}`, cb.x + cb.w - 4, sy(v(k)) - 6, { size: 11, weight: 650, anchor: 'end', halo: true });
    });
    const rx = nw ? pm : cb.x + cb.w + 28, rw = nw ? W - 2 * pm : W - rx - pm, ry = nw ? cb.y + cb.h + 56 : 10;
    text(E, 'One sampled answer to a 4-digit problem', rx, ry + 4, { size: nw ? 11 : 12, fill: MUTED });
    sampleCard(E, { x: rx, y: ry + 12, w: rw, h: 60 }, liveSample(r, f.it, 4, 'strict'), 'Sampled');
  },
  label: (r) => `Mean response length in words for 1 to 4 digit problems over ${r.iters} training iterations, with a sampled answer.`,
};
export default mode;
