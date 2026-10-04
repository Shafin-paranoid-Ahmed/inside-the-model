/**
 * qlearn-terminal (q-learning step 5): three real Q-learning updates (from a recorded run with gamma 0.9) side by side,
 * one that ends in the goal, one that ends in the pit and one ordinary move, each with the same target card:
 * target = r + gamma (1 - terminal flag) max Q(s'). Cards build row by row. Props: transitions ("goal,pit,ordinary"),
 * alpha (0.1), gamma (0.9), epsilon (0.3), seed (11), episode (50: pick the first matching update from that episode on).
 */
import { ACTIONS, GOAL, cellName } from '../../../lib/rl/gridworld';
import { ACCENT, MUTED, RULE, ease, fmt, rect, signedFill, text, type Env } from '../_rl/svg';
import { int, list, num, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { gridOf } from './sim';
import { qRun } from './q-common';

type Kind = 'goal' | 'pit' | 'ordinary';
const TITLE: Record<Kind, string> = { goal: 'Into the goal', pit: 'Into the pit', ordinary: 'An ordinary move' };
const LEAD = 0.5, ROW = 0.75, ROWS = 5, SEG = LEAD + ROWS * ROW + 1.2;
interface Card { kind: Kind; s: number; a: number; s2: number; r: number; done: boolean; maxNext: number; target: number; before: number; err: number; after: number; ep: number }
interface Scene { cards: Card[]; alpha: number; gamma: number }
interface F extends Frame { card: number; rows: number[] }

const mode: ModeImpl<Scene, F> = {
  setup(o: Opts, kit) {
    const g = gridOf(o, 0.9), from = int(o.episode, 50), alpha = num(o.alpha, 0.1);
    const run = qRun(kit, g, { ...o, gamma: g.gamma }, { alpha, epsilon: 0.3, episodes: from + 40, seed: 11 }), tr = run.trace!;
    const kinds = list(o.transitions, ['goal', 'pit', 'ordinary']).filter((k): k is Kind => k in TITLE);
    const cards: Card[] = kinds.map((kind) => {
      let i = tr.ep.findIndex((e) => e >= from - 1);
      for (; i < tr.s.length; i++) {
        const done = tr.done[i] === 1;
        if (kind === 'goal' ? done && tr.s2[i] === GOAL : kind === 'pit' ? done && tr.s2[i] !== GOAL : !done && tr.s2[i] === 10 && tr.s[i] !== 10) break;
      }
      i = Math.min(i, tr.s.length - 1);
      const done = tr.done[i] === 1, r = tr.r[i], target = tr.target[i];
      return { kind, s: tr.s[i], a: tr.a[i], s2: tr.s2[i], r, done, maxNext: done ? 0 : (target - r) / g.gamma, target, before: tr.qBefore[i], err: tr.tdError[i], after: tr.qBefore[i] + alpha * tr.tdError[i], ep: tr.ep[i] + 1 };
    });
    return { cards, alpha, gamma: g.gamma };
  },
  timeline(s) {
    return { duration: s.cards.length * SEG + 1, markers: s.cards.map((c, i) => ({ at: i * SEG, label: TITLE[c.kind] })) };
  },
  frameAt(t, s) {
    const n = s.cards.length, i = Math.min(n - 1, Math.floor(t / SEG)), local = t - i * SEG, c = s.cards[i];
    const rows = Array.from({ length: ROWS }, (_, j) => ease((local - LEAD - j * ROW) / 0.4));
    const where = `${cellName(c.s)} ${ACTIONS[c.a]}, landing in ${c.done ? (c.s2 === GOAL ? 'the goal' : 'the pit') : cellName(c.s2)}`;
    const stage = Math.min(ROWS - 1, Math.max(0, Math.floor((local - LEAD) / ROW)));
    let status: string;
    if (local < LEAD) status = `Take one recorded update from episode ${c.ep}: ${where}.`;
    else if (stage === 0) status = `The move paid a reward of ${fmt(c.r)}${c.done ? (c.r > 0 ? ': +1 for the goal minus the 0.04 move cost' : ': −1 for the pit minus the 0.04 move cost') : ': just the move cost'}.`;
    else if (stage === 1) status = c.done ? 'The episode is over, so the terminal flag is 1.' : 'The episode goes on, so the terminal flag is 0.';
    else if (stage === 2) status = c.done ? 'With the flag at 1 the next-state term is multiplied by 0: there is no next action to value.' : `The best Q in ${cellName(c.s2)} is ${fmt(c.maxNext, 3)}: what the agent expects to collect from there.`;
    else if (stage === 3) status = c.done ? `The target is the reward alone: ${fmt(c.target, 3)}.` : `The target is ${fmt(c.r)} + ${fmt(s.gamma, 1)} × ${fmt(c.maxNext, 3)} = ${fmt(c.target, 3)}.`;
    else status = `The old estimate was ${fmt(c.before, 3)}, so the error is ${fmt(c.err, 3)} and Q moves ${fmt(s.alpha, 1)} of the way to ${fmt(c.after, 3)}.`;
    return { card: i, rows, status, readouts: [['Reward', fmt(c.r)], ['Terminal flag', c.done ? '1' : '0'], ['Target', fmt(c.target, 3)], ['New Q', fmt(c.after, 3)]] };
  },
  draw(E: Env, f, s) {
    const n = s.cards.length, nar = E.narrow, gap = 12;
    const cw = nar ? E.W - 16 : (E.W - 16 - gap * (n - 1)) / n, ch = nar ? (E.H - 8 - gap * (n - 1)) / n : E.H - 16;
    s.cards.forEach((c, i) => {
      if (i > f.card) return;
      const x = 8 + (nar ? 0 : i * (cw + gap)), y = 8 + (nar ? i * (ch + gap) : 0), active = i === f.card, pad = 12, rh = Math.min(26, (ch - 56) / 7);
      rect(E, x, y, cw, ch, { fill: 'none', stroke: active ? ACCENT : RULE, sw: active ? 2 : 1, rx: 6 });
      text(E, TITLE[c.kind], x + pad, y + 20, { size: 13, weight: 650 });
      text(E, `${cellName(c.s)} ${ACTIONS[c.a]} → ${c.done ? (c.s2 === GOAL ? 'goal' : 'pit') : cellName(c.s2)}`, x + pad, y + 38, { size: 12, fill: MUTED });
      const vis = i < f.card ? [1, 1, 1, 1, 1] : f.rows;
      const row = (j: number, k: string, v: string, o: { fill?: string; bold?: boolean } = {}) => {
        const yy = y + 62 + j * rh;
        text(E, k, x + pad, yy, { size: 12, fill: MUTED, op: vis[Math.min(4, Math.round(j))] });
        text(E, v, x + cw - pad, yy, { size: 12, anchor: 'end', mono: true, weight: o.bold ? 650 : 400, fill: o.fill, op: vis[Math.min(4, Math.round(j))] });
      };
      row(0, 'reward r', fmt(c.r));
      row(1, 'terminal flag', c.done ? '1  (over)' : '0  (goes on)');
      row(2, 'max Q(s′)', c.done ? 'not used' : fmt(c.maxNext, 3));
      row(3, 'target', fmt(c.target, 3), { bold: true });
      text(E, c.done ? `${fmt(c.r)}  + 0 from the next state` : `${fmt(c.r)} + ${fmt(s.gamma, 1)} × ${fmt(c.maxNext, 3)}`, x + pad, y + 62 + 3.85 * rh, { size: 11, fill: MUTED, mono: true, op: vis[3] });
      row(4.9, 'old Q → new Q', `${fmt(c.before, 3)} → ${fmt(c.after, 3)}`);
      // a signed swatch of the target so goal, pit and ordinary targets can be compared at a glance
      rect(E, x + pad, y + ch - 14, cw - 2 * pad, 6, { fill: signedFill(c.target, 1), op: vis[4], rx: 3 });
    });
  },
  label: (s) => `Three recorded Q-learning updates, ${s.cards.map((c) => TITLE[c.kind].toLowerCase()).join(', ')}, each with its reward, terminal flag, next-state value, target and updated Q.`,
};
export default mode;
