/**
 * rm-bias (rlhf step 3): every candidate response of every prompt (80 points) as (gold quality, reward model score),
 * coloured by candidate type. The seven ordinary types appear one type at a time, a line is fitted through them, and
 * then the rambling wrong answer (48 words, longer than anything in the pairs, whose longest response is 28 words) is
 * circled: the linear reward model gives it the highest score of all.
 * Props: labels ("human"), pairs (200), seed (4 = pair seed).
 * Measured (seed 4): Spearman rank correlation of RM with gold 0.871 on the 68 candidates within the lengths seen in
 * pairs (plan §7; this mode splits rambling vs the rest instead, see the test), 0.478 over all 80; the top RM candidate
 * is rambling wrong, RM 4.72, gold -0.44.
 */
import { CANDIDATES, K, P, PROMPTS, Y_RAMBLING, phi, F_LENGTH } from '../../../lib/align/bank';
import { GOLD, scores } from '../../../lib/align/policy';
import { int, type Frame, type Kit, type ModeImpl, type Opts } from '../_rl/kit';
import { ACCENT, BG, MUTED, axes, circle, clamp, ease, fmt, line, text, type Env } from '../_rl/svg';
import { candColor } from '../_align/draw';
import { prefData } from './sim';
import { spearman } from './rlhf-common';

const LEAD = 0.6, GROUP = 1.0, TREND = 1.8, HI = 2.2, HOLD = 3;
interface Pt { gold: number; rm: number; y: number; x: number; len: number }
interface Scene { pts: Pt[]; slope: number; icpt: number; rhoRest: number; rhoAll: number; top: Pt; maxSeen: number; lo: [number, number]; hi: [number, number] }
interface F extends Frame { groups: number; fade: number; trend: number; hi: number; phase: 0 | 1 | 2 }

function bias(o: Opts, kit: Kit): Scene {
  const d = prefData(o, kit, int(o.seed, 4)), sc = scores(d.rm.w);
  const pts: Pt[] = [];
  for (let x = 0; x < P; x++) for (let y = 0; y < K; y++) pts.push({ gold: GOLD[x * K + y], rm: sc[x * K + y], y, x, len: Math.round(phi(x, y)[F_LENGTH] * 10) });
  let maxSeen = 0;
  for (const q of d.pairs) for (const y of [q.a, q.b]) maxSeen = Math.max(maxSeen, Math.round(phi(q.x, y)[F_LENGTH] * 10));
  const rest = pts.filter((p) => p.y !== Y_RAMBLING), n = rest.length;
  const mg = rest.reduce((s, p) => s + p.gold, 0) / n, mr = rest.reduce((s, p) => s + p.rm, 0) / n;
  let sgg = 0, sgr = 0;
  for (const p of rest) { sgg += (p.gold - mg) ** 2; sgr += (p.gold - mg) * (p.rm - mr); }
  const slope = sgr / sgg, icpt = mr - slope * mg, top = pts.reduce((a, b) => (b.rm > a.rm ? b : a));
  return {
    pts, slope, icpt, top, maxSeen, rhoRest: spearman(rest.map((p) => p.gold), rest.map((p) => p.rm)), rhoAll: spearman(pts.map((p) => p.gold), pts.map((p) => p.rm)),
    lo: [Math.min(...pts.map((p) => p.gold)) - 0.3, Math.min(...pts.map((p) => p.rm)) - 0.6], hi: [Math.max(...pts.map((p) => p.gold)) + 0.3, Math.max(...pts.map((p) => p.rm)) + 0.6],
  };
}

const mode: ModeImpl<Scene, F> = {
  setup: (o, kit) => bias(o, kit),
  timeline: () => ({
    duration: LEAD + K * GROUP + TREND + HI + HOLD,
    markers: [{ at: 0, label: 'Points by type' }, { at: LEAD + K * GROUP, label: 'Trend line' }, { at: LEAD + K * GROUP + TREND, label: 'Longest answer' }],
  }),
  frameAt(t, s) {
    const g = clamp(Math.floor((t - LEAD) / GROUP) + 1, 0, K), fade = ease(((t - LEAD) % GROUP) / 0.4), tEnd = LEAD + K * GROUP;
    const trend = ease((t - tEnd) / (TREND - 0.4)), hi = ease((t - tEnd - TREND) / 0.8), phase = t < tEnd ? 0 : t < tEnd + TREND ? 1 : 2;
    let status: string;
    if (phase === 0) {
      const y = Math.max(0, g - 1), mine = s.pts.filter((p) => p.y === y), r = (k: 'gold' | 'rm') => [Math.min(...mine.map((p) => p[k])), Math.max(...mine.map((p) => p[k]))];
      const [g0, g1] = r('gold'), [r0, r1] = r('rm'), span = (a: number, b: number) => (Math.abs(a - b) < 0.005 ? fmt(a, 2) : `${fmt(a, 2)} to ${fmt(b, 2)}`);
      status = t < LEAD ? 'Every response in the bank gets a gold score (what a person really wanted) and a reward model score. Each dot is one response to one prompt.' : `${CANDIDATES[y].label}: gold ${span(g0, g1)}, reward model ${span(r0, r1)}. There are ${mine.length} of them, one per prompt.`;
    } else if (phase === 1) status = `The seven types so far lie close to a rising line: a higher gold score usually means a higher reward model score (rank correlation ${fmt(s.rhoRest, 2)}). The rambling wrong answer has not been shown yet.`;
    else status = `The rambling wrong answer (${s.top.len} words, the longest response in the pairs was ${s.maxSeen}) scores ${fmt(s.top.rm, 2)}, higher than any other response, although its gold score is ${fmt(s.top.gold, 2)}. With it included the rank correlation falls to ${fmt(s.rhoAll, 2)}.`;
    return {
      groups: g, fade, trend, hi, phase, status,
      readouts: [['Rank correlation, other seven types', phase ? fmt(s.rhoRest, 2) : '–'], ['Rank correlation, all eight', phase === 2 ? fmt(s.rhoAll, 2) : '–'], ['Highest reward model score', phase === 2 ? `${fmt(s.top.rm, 2)} (gold ${fmt(s.top.gold, 2)})` : '–']],
    };
  },
  draw(E: Env, f, s) {
    const nw = E.narrow, b = { x: nw ? 40 : 54, y: 30, w: E.W - (nw ? 40 : 54) - 14, h: E.H - 30 - 50 };
    const { sx, sy } = axes(E, b, { lo: s.lo[0], hi: s.hi[0], ticks: [-1, 0, 1, 2, 3], label: 'Gold score (what a person really wanted)' }, { lo: s.lo[1], hi: s.hi[1], ticks: [-2, 0, 2, 4], label: 'Reward model score' });
    if (f.trend > 0) {
      const xa = s.lo[0], xb = xa + (s.hi[0] - xa) * f.trend;
      line(E, sx(xa), sy(s.icpt + s.slope * xa), sx(xb), sy(s.icpt + s.slope * xb), { stroke: MUTED, sw: 2, dash: '6 4' });
    }
    const focus = f.phase === 2;
    for (const p of s.pts) {
      const ramble = p.y === Y_RAMBLING;
      if (p.y >= f.groups && !(f.phase === 2)) continue;
      if (ramble && f.phase < 2) continue;
      const op = (p.y === f.groups - 1 && f.phase === 0 ? 0.25 + 0.75 * f.fade : 1) * (focus && !ramble ? 1 - 0.55 * f.hi : 1);
      circle(E, sx(p.gold), sy(p.rm), 5.5, { fill: candColor(p.y), stroke: BG, sw: 1, op, tip: `“${PROMPTS[p.x].q}” ${CANDIDATES[p.y].label.toLowerCase()}: gold ${fmt(p.gold, 2)}, reward model ${fmt(p.rm, 2)}` });
    }
    if (focus) {
      circle(E, sx(s.top.gold), sy(s.top.rm), 5.5 + 7 * f.hi, { fill: 'none', stroke: ACCENT, sw: 2.5, op: f.hi });
      const lab = nw ? 'Rambling wrong' : `Rambling wrong, ${s.top.len} words: reward ${fmt(s.top.rm, 2)}, gold ${fmt(s.top.gold, 2)}`;
      text(E, lab, sx(s.top.gold) + 14, sy(s.top.rm) + 4, { size: 12, weight: 600, halo: true, op: f.hi, anchor: sx(s.top.gold) + 14 + lab.length * 6.4 > E.W ? 'end' : 'start' });
    }
  },
  label: (s) => `Scatter of ${s.pts.length} candidate responses, reward model score against gold score. The rambling wrong answer has the highest reward model score (${fmt(s.top.rm, 2)}) and a gold score of ${fmt(s.top.gold, 2)}.`,
};
export default mode;
