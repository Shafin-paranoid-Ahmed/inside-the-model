/**
 * reinforce-update (policy-gradients step 4): one REINFORCE episode turned into one policy update. The agent walks the
 * episode with the policy held fixed; returns G_t fill in backward; the learned state-value baseline b(s_t) is subtracted
 * to give advantages; the pushes lr * A_t * (onehot(a_t) - pi(s_t)) are added up for the chosen cell; the policy changes once.
 * Props: seed (27), episode (10, as numbered on screen), lr (0.1), focus ("3,2"), baseline ("state-value" | "none").
 * Measured (seed 27, episode 10): 8 steps into the pit; cell (3,2) is visited once with G -1.04, b -0.49, advantage -0.55.
 */
import { ACTIONS, ARROWS, GOAL, PIT, cellName, parseCell, xy } from '../../../lib/rl/gridworld';
import { gridReinforce, softmaxRow, type PGEpisodeDetail } from '../../../lib/rl/tabular';
import { int, num, str, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { ACCENT, MUTED, RULE, TEXT, ease, fmt, inkOn, lerp, line, rect, signFmt, signedFill, text, type Env } from '../_rl/svg';
import { cellXY, drawAgent, drawGrid, gridIn } from './draw';
import { gridOf } from './sim';

const LEAD = 0.5, MOVE = 0.6, RET = 2.4, BASE = 3, GRAD = 2.6, UPD = 2.6, HOLD = 2.2;
interface Scene {
  d: PGEpisodeDetail; T: number; ep: number; lr: number; focus: number; fin: number; path: number[];
  before: Float64Array; after: Float64Array; push: Float64Array; taken: number[]; visits: number[]; key: number;
  t: { c1: number; c2: number; c3: number; c4: number; end: number };
}
interface F extends Frame { t: number; phase: number; k: number; u: number; fx: number; fy: number }

const num4 = (x: number) => fmt(x, 2);

const mode: ModeImpl<Scene, F> = {
  setup(o: Opts) {
    const g = gridOf(o), ep = int(o.episode, 10), lr = num(o.lr, 0.1), base = str(o.baseline, ['state-value', 'none'] as const, 'state-value');
    const res = gridReinforce(g, { lr, episodes: ep, seed: int(o.seed, 27), baseline: base, lrCritic: 0.1, evalEvery: 0, detail: [ep - 1] });
    const d = res.details[0], T = d.steps.length, focus = parseCell(o.focus, 6);
    const fin = d.steps[T - 1].r > 0.5 ? GOAL : PIT;
    const path = [...d.steps.map((s) => s.s), fin];
    const before = softmaxRow(d.thetaBefore, focus), after = softmaxRow(d.thetaAfter, focus), push = new Float64Array(4);
    for (let a = 0; a < 4; a++) push[a] = lr * d.grad[focus * 4 + a];
    const taken = [0, 0, 0, 0], visits: number[] = [];
    d.steps.forEach((s, j) => { if (s.s === focus) { taken[s.a]++; visits.push(j); } });
    let key = 0;
    for (let a = 1; a < 4; a++) if (Math.abs(push[a]) > Math.abs(push[key])) key = a;
    const c1 = LEAD + T * MOVE, c2 = c1 + RET, c3 = c2 + BASE, c4 = c3 + GRAD, end = c4 + UPD;
    return { d, T, ep, lr, focus, fin, path, before, after, push, taken, visits, key, t: { c1, c2, c3, c4, end } };
  },
  timeline: (s) => ({
    duration: s.t.end + HOLD,
    markers: [{ at: 0, label: 'Play the episode' }, { at: s.t.c1, label: 'Add up the returns' }, { at: s.t.c2, label: 'Subtract the baseline' }, { at: s.t.c3, label: 'Add up the pushes' }, { at: s.t.c4, label: 'Update the policy' }],
  }),
  frameAt(t, s) {
    const { d, T, t: tt } = s, phase = t < tt.c1 ? 0 : t < tt.c2 ? 1 : t < tt.c3 ? 2 : t < tt.c4 ? 3 : 4;
    const raw = (t - LEAD) / MOVE, k = Math.max(0, Math.min(T - 1, Math.floor(raw))), u = phase > 0 || raw >= T ? 1 : Math.max(0, raw - k);
    const [x0, y0] = xy(s.path[k]), [x1, y1] = xy(s.path[k + 1]), m = phase > 0 ? 1 : t < LEAD ? 0 : ease(u / 0.7);
    const kk = phase > 0 ? T - 1 : k, fx = phase > 0 ? xy(s.fin)[0] : x0 + (x1 - x0) * m, fy = phase > 0 ? xy(s.fin)[1] : y0 + (y1 - y0) * m;
    const name = cellName(s.focus), v = s.visits[0], A = (a: number) => ARROWS[a] + ' ' + ACTIONS[a];
    const stepAt = d.steps[kk];
    let status: string;
    if (phase === 0) status = t < LEAD ? `The policy is held fixed while the agent plays one episode, picking every action with the policy's own probabilities.` : `Step ${kk + 1}: at ${cellName(stepAt.s)} it picks ${ACTIONS[stepAt.a]}. Reward ${fmt(stepAt.r)}.`;
    else if (phase === 1) status = `Returns, worked backward from the end: the last step's return G is its reward ${fmt(d.steps[T - 1].r)}, and each earlier G is its reward plus the G after it. The first step's G is ${fmt(d.steps[0].G)}.`;
    else if (phase === 2) status = t < tt.c2 + BASE / 2
      ? `The baseline b is the return this policy has been getting from each cell so far, learned over the episodes before this one.`
      : v === undefined ? `Advantage = G minus b: how much better this episode went than usual from that cell. The episode never visited ${name}.`
        : `Advantage = G minus b. At ${name}: ${fmt(d.steps[v].G)} minus ${fmt(d.steps[v].b)} is ${signFmt(d.steps[v].adv)}, so it went ${d.steps[v].adv >= 0 ? 'better' : 'worse'} than usual from there.`;
    else if (phase === 3) status = v === undefined ? `${name} was not visited, so the episode pushes none of its logits.`
      : `Each visit to ${name} pushes its four logits by ${s.lr} × advantage × (1 for the action taken, 0 for the others, minus that action's probability). Total push on ${ACTIONS[s.key]}: ${signFmt(s.push[s.key], 3)}.`;
    else status = v === undefined ? `Nothing changes at ${name}.` : t < tt.c4 + UPD * 0.35
      ? `The logits move by the pushes, and the policy changes once for the whole episode.`
      : `At ${name} the chance of ${ACTIONS[s.key]} goes from ${num4(s.before[s.key])} to ${num4(s.after[s.key])}.`;
    const settled = phase === 4 && t >= tt.c4 + UPD * 0.35;
    return {
      t, phase, k: kk, u, fx, fy, status,
      detail: `Episode ${s.ep}, ${T} steps, ends in the ${s.fin === GOAL ? '+1 goal' : '−1 pit'}. Choose another cell under the figure to see its update.`,
      readouts: [['Cell shown', name], ['Visits in this episode', String(s.visits.length)], [`Chance of ${A(s.key)}`, num4(settled ? s.after[s.key] : s.before[s.key])]],
    };
  },
  draw(E: Env, f, s) {
    const { d, T, t: tt } = s, nar = E.narrow, W = E.W, rows = 7, rh = 18, stripH = rows * rh + 8;
    const top = { x: 8, y: 22, w: W - 16, h: E.H - stripH - 38 };
    const gb = nar ? { ...top, h: Math.min(top.h * 0.5, 230) } : { ...top, w: W * 0.38 };
    text(E, `Episode ${s.ep}`, gb.x + 4, 14, { fill: MUTED, weight: 600 });
    const G = gridIn(gb), visited = new Set(s.path.slice(0, T));
    drawGrid(E, G, { highlight: s.focus, tip: (c) => (visited.has(c) ? `${cellName(c)}: visited in this episode` : cellName(c)) });
    // path travelled so far
    const reached = f.phase > 0 ? T : f.k + f.u;
    for (let j = 0; j < T; j++) {
      if (j >= reached || s.path[j] === s.path[j + 1]) continue;
      const a = cellXY(G, s.path[j]), b = cellXY(G, s.path[j + 1]), p = Math.min(1, reached - j);
      line(E, a.cx, a.cy, a.cx + (b.cx - a.cx) * p, a.cy + (b.cy - a.cy) * p, { stroke: 'var(--viz-forward)', sw: 2.5, op: 0.7 });
    }
    drawAgent(E, G, f.fx, f.fy);
    // the chosen cell's policy
    const P = nar ? { x: 8, y: gb.y + gb.h + 12, w: W - 16, h: top.h - gb.h - 12 } : { x: gb.x + gb.w + 22, y: 22, w: W - gb.w - 46, h: top.h };
    const lw = nar ? 62 : 84, cw = (P.w - lw) / 3, sz = nar ? 11 : 12, ry0 = P.y + 40, rh2 = Math.min(40, (P.h - 44) / 4);
    text(E, `Cell ${cellName(s.focus)}: the four action probabilities`, P.x, P.y - 8 + 6, { weight: 600, size: sz });
    const heads = ['Chance before', 'Push on logit', 'Chance after'], showPush = f.phase >= 3, showAfter = f.phase >= 4;
    heads.forEach((h, j) => text(E, h, P.x + lw + cw * j + cw / 2, P.y + 26, { anchor: 'middle', fill: MUTED, size: 11 }));
    const maxPush = Math.max(0.1, ...Array.from(s.push, (x) => Math.abs(x) * 1.15));
    const pushGrow = f.phase === 3 ? ease((f.t - tt.c3) / (GRAD * 0.6)) : 1, afterU = ease((f.t - tt.c4) / (UPD * 0.35)), settled = f.t >= tt.c4 + UPD * 0.35;
    for (let a = 0; a < 4; a++) {
      const y = ry0 + a * rh2;
      text(E, `${ARROWS[a]} ${ACTIONS[a]}`, P.x, y + 14, { size: sz, weight: s.taken[a] ? 650 : 400 });
      if (s.taken[a]) text(E, `taken ${s.taken[a]}×`, P.x, y + 28, { size: 11, fill: MUTED });
      const bw = cw - 34;
      rect(E, P.x + lw, y + 4, bw, 14, { fill: 'none', stroke: RULE, sw: 1 });
      rect(E, P.x + lw, y + 4, bw * s.before[a], 14, { fill: 'var(--viz-forward)', op: 0.85 });
      text(E, num4(s.before[a]), P.x + lw + bw + 4, y + 16, { size: 11 });
      const cx = P.x + lw + cw + cw / 2 - 8, hw = cw / 2 - 24;
      line(E, cx, y + 2, cx, y + 20, { stroke: RULE, sw: 1 });
      if (showPush) {
        const g = s.push[a], w = (g / maxPush) * hw * pushGrow;
        rect(E, Math.min(cx, cx + w), y + 4, Math.abs(w), 14, { fill: 'var(--viz-gradient)', op: 0.9 });
        text(E, signFmt(g, 3), g >= 0 ? cx - 3 : cx + 3, y + 30, { size: 11, anchor: g >= 0 ? 'end' : 'start', fill: MUTED });
      }
      if (showAfter) {
        const x0 = P.x + lw + cw * 2;
        rect(E, x0, y + 4, bw, 14, { fill: 'none', stroke: RULE, sw: 1 });
        rect(E, x0, y + 4, bw * lerp(s.before[a], s.after[a], afterU), 14, { fill: 'var(--viz-forward)', op: 0.85 });
        line(E, x0 + bw * s.before[a], y + 1, x0 + bw * s.before[a], y + 21, { stroke: TEXT, sw: 1.5, dash: '3 2' });
        if (settled) text(E, num4(s.after[a]), x0 + bw + 4, y + 16, { size: 11 });
      }
    }
    // the episode as a table, one column per step
    const sy = E.H - stripH + 2, lx = 8, tx = lx + (nar ? 62 : 84), maxCols = Math.floor((W - tx - 6) / (nar ? 32 : 38)), nC = Math.min(T, maxCols), cwid = Math.min(54, (W - tx - 6) / nC);
    const labels = ['Step', 'Cell', 'Action', 'Reward', 'Return G', 'Baseline b', 'Advantage'];
    labels.forEach((l, r) => text(E, nar && l.length > 9 ? l.replace('Baseline b', 'Base b') : l, lx, sy + r * rh + 13, { size: 11, fill: MUTED }));
    const vmax = Math.max(0.5, ...d.steps.map((x) => Math.abs(x.adv))), shownMoves = f.phase > 0 ? T : f.k + (f.u >= 0.7 ? 1 : 0);
    for (let j = 0; j < nC; j++) {
      const st = d.steps[j], x = tx + j * cwid + cwid / 2;
      if (st.s === s.focus && j < shownMoves) rect(E, x - cwid / 2 + 1, sy - 2, cwid - 2, stripH - 4, { fill: 'none', stroke: ACCENT, sw: 1.5, rx: 3 });
      if (j >= shownMoves) continue;
      const cell = (r: number, str: string, o: { fill?: string; bg?: string } = {}) => {
        if (o.bg) rect(E, x - cwid / 2 + 2, sy + r * rh + 1, cwid - 4, rh - 2, { fill: o.bg, rx: 2 });
        text(E, str, x, sy + r * rh + 13, { size: 11, anchor: 'middle', fill: o.fill ?? TEXT, mono: true });
      };
      cell(0, String(j + 1)); cell(1, cellName(st.s).replace(/[()]/g, '')); cell(2, ARROWS[st.a]); cell(3, fmt(st.r));
      if (f.t >= tt.c1 + ((T - 1 - j) / T) * RET * 0.85) cell(4, fmt(st.G));
      if (f.t >= tt.c2) cell(5, fmt(st.b));
      if (f.t >= tt.c2 + BASE / 2) cell(6, signFmt(st.adv), { bg: signedFill(st.adv, vmax), fill: inkOn(Math.abs(st.adv) / vmax) });
    }
    if (nC < T) text(E, `… ${T - nC} more steps`, W - 8, sy + 13, { size: 11, fill: MUTED, anchor: 'end' });
  },
  label: (s) => `One REINFORCE episode of ${s.T} steps turned into one policy update: returns, baselines, advantages, the pushes on the logits of cell ${cellName(s.focus)}, and its action probabilities before and after.`,
};
export default mode;

