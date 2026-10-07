/**
 * arena (evaluations step 2): pairs of anonymous models get one prompt, a voter picks the better answer, and a
 * Bradley-Terry fit turns the vote counts into ratings with Fisher-information intervals.
 * Props: votes (3000), seed (3), lengthBias (toggle: voters add a bonus for the long-winded Model C).
 * Both vote streams use the same pairs and the same coin flips; only the chance of each win differs.
 * Left of the ratings, every 6th of the most recent votes flies to the row of the model that won it.
 */
import { bool, int, key, type Frame, type Kit, type ModeImpl, type Opts } from '../_rl/kit';
import { BG, MUTED, RULE, TEXT, circle, fmt, line, path, rect, text, type Env } from '../_rl/svg';
import { ARENA_BETA, ARENA_LEN, MODEL_NAMES, arenaAt, catColor, makeArena, toRating, type Arena } from './sim';

const PLAY = 14, HOLD = 3, MIN = 30, WINDOW = 144, EVERY = 6;
const WORDY = ARENA_LEN.indexOf(1);
interface Scene { ar: Arena; seed: number; truth: number[]; trueRank: number[] }
interface F extends Frame { n: number; nf: number; r: ReturnType<typeof arenaAt>; biased: boolean; wins: number[] }
const fmtN = (n: number) => n.toLocaleString('en-US');

const mode: ModeImpl<Scene, F> = {
  setup(o: Opts, kit: Kit) {
    const votes = int(o.votes, 3000), seed = int(o.seed, 3);
    const ar = kit.memo(key('arena', seed, votes), () => makeArena(seed, votes));
    const truth = ARENA_BETA.map(toRating), order = truth.map((r, i) => [r, i]).sort((a, b) => b[0] - a[0]).map((x) => x[1]), trueRank = new Array<number>(truth.length);
    order.forEach((m, r) => { trueRank[m] = r + 1; });
    return { ar, seed, truth, trueRank };
  },
  timeline: (s) => ({
    duration: PLAY + HOLD,
    markers: [{ at: 0, label: 'First 300 votes' }, { at: (PLAY * 300) / s.ar.votes, label: '300 to 1,000' }, { at: (PLAY * 1000) / s.ar.votes, label: '1,000 to 3,000' }],
  }),
  frameAt(t, s, o) {
    const V = s.ar.votes, biased = bool(o.lengthBias, false), nf = Math.max(MIN, Math.min(V, (t / PLAY) * V)), n = Math.floor(nf);
    const r = arenaAt(s.ar, n, biased), c = WORDY;
    const lo = r.rating.indexOf(Math.min(...r.rating)), hi = r.rating.indexOf(Math.max(...r.rating)), half = r.half.reduce((a, b) => a + b, 0) / r.half.length;
    const d = Math.round(r.rating[c] - s.truth[c]), status = biased
      ? `After ${fmtN(n)} votes, ${MODEL_NAMES[c]} (the wordy one) is ranked #${r.rank[c]} at ${fmt(r.rating[c], 0)}, ${Math.abs(d)} points ${d >= 0 ? 'above' : 'below'} the ${fmt(s.truth[c], 0)} that fair voters give it. Its answers are no more correct; they are just longer.`
      : `After ${fmtN(n)} votes the ratings run from ${fmt(r.rating[lo], 0)} (${MODEL_NAMES[lo]}) to ${fmt(r.rating[hi], 0)} (${MODEL_NAMES[hi]}), and each interval is about plus or minus ${fmt(half, 0)} points.`;
    const wm = (biased ? s.ar.winsBiased : s.ar.winsFair).subarray(n * s.ar.k * s.ar.k, (n + 1) * s.ar.k * s.ar.k), wins = Array.from({ length: s.ar.k }, (_, i) => { let c = 0; for (let j = 0; j < s.ar.k; j++) c += wm[i * s.ar.k + j]; return c; });
    return { n, nf, r, biased, wins, status, readouts: [['Votes', fmtN(n)], [MODEL_NAMES[c], `${fmt(r.rating[c], 0)} ± ${fmt(r.half[c], 0)}, rank ${r.rank[c]}`]] };
  },
  draw(E: Env, f, s) {
    const nar = E.narrow, k = s.ar.k, top = 28, rh = (E.H - top - 40) / k, labelW = nar ? 74 : 104, laneW = nar ? 44 : Math.min(170, E.W * 0.22), lx = labelW, fx = lx + laneW + (nar ? 14 : 30), fw = E.W - fx - 14;
    const lo = 750, hi = 1250, sx = (v: number) => fx + ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * fw, rowY = (m: number) => top + (m + 0.5) * rh;
    text(E, nar ? 'Votes' : 'Recent votes, loser to winner', lx, top - 12, { size: 11, fill: MUTED });
    text(E, nar ? 'Rating' : 'Rating, with 95% interval', fx, top - 12, { size: 11, fill: MUTED });
    for (const v of [800, 1000, 1200]) {
      line(E, sx(v), top, sx(v), top + k * rh, { stroke: RULE, sw: 1, dash: '2 4' });
      text(E, String(v), sx(v), top + k * rh + 15, { size: 11, anchor: 'middle', fill: MUTED });
    }
    for (let m = 0; m < k; m++) {
      const y = rowY(m);
      rect(E, 0, y - 12, 8, 8, { fill: catColor(m) });
      text(E, MODEL_NAMES[m], 13, y - 3, { weight: 600, size: 12 });
      text(E, `#${f.r.rank[m]} · ${fmt(f.r.rating[m], 0)}`, 13, y + 12, { size: 11, fill: MUTED });
      text(E, `${f.wins[m].toLocaleString('en-US')} wins`, 13, y + 24, { size: 11, fill: MUTED });
      line(E, sx(s.truth[m]), y - 11, sx(s.truth[m]), y + 11, { stroke: TEXT, sw: 1.75, dash: '2 2' });
      line(E, sx(f.r.rating[m] - f.r.half[m]), y, sx(f.r.rating[m] + f.r.half[m]), y, { stroke: catColor(m), sw: 4.5, op: 0.55 });
      circle(E, sx(f.r.rating[m]), y, 4.5, { fill: catColor(m), stroke: BG, sw: 1.5 });
    }
    const vs = s.ar;
    for (let v = Math.floor(f.nf) - 1; v >= Math.max(0, f.n - WINDOW); v--) {
      if (v % EVERY) continue;
      const p = (f.nf - v) / WINDOW, w = f.biased ? vs.biased[v] : vs.fair[v], l = w === vs.a[v] ? vs.b[v] : vs.a[v];
      const x0 = lx + 6, yl = rowY(l), yw = rowY(w), bulge = Math.min(laneW - 10, 14 + Math.abs(yw - yl) * 0.22);
      path(E, `M${x0.toFixed(1)} ${yl.toFixed(1)}Q${(x0 + bulge * 2).toFixed(1)} ${((yl + yw) / 2).toFixed(1)} ${x0.toFixed(1)} ${yw.toFixed(1)}`, { stroke: catColor(w), sw: 1.6, op: 0.85 * (1 - p) });
      circle(E, x0, yw, 3.2, { fill: catColor(w), op: 0.95 * (1 - p) });
    }
    if (!nar) text(E, 'Dashed tick: the rating the model deserves if voters judge only quality', fx + fw, top + k * rh + 31, { size: 11, anchor: 'end', fill: MUTED });
  },
  label: (s) => `Pairwise votes between six fictional models become Bradley-Terry ratings with 95% intervals after ${s.ar.votes} votes; a toggle makes voters favour longer answers.`,
  restTime: () => PLAY,
};
export default mode;
