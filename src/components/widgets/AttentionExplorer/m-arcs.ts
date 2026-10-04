/** The arcs figure: its frame function and its canvas drawing. */
import { ease, f2, type Scene, ARC_SEG, type ArcsFrame, type Opts, seg, easedRow, topWord, pre, type Frame } from './core';
import { type Env } from './ui';
import { paintArcs } from './arcs';

export function frameAt(t: number, s: Scene, o: Opts): Frame {
  const { tt, T } = pre(t, s, o);
  const { i, u } = o.pin != null ? { i: o.pin, u: 1 } : seg(tt, ARC_SEG, T);
  // The previous word's arcs fade out while the new word's real arcs fade in; name, numbers and narration switch at once.
  const target = easedRow(s, i, 1), prev = i > 0 ? easedRow(s, i - 1, 1) : new Float64Array(T), fade = ease(u / 0.15), top = topWord(s, target);
  const status = i === 0 ? `“${s.words[0]}” is first, so it can only look at itself.` :
    `“${s.words[i]}” puts ${f2(top.w)} of its attention on “${top.word}”.`;
  return { mode: 'arcs', q: i, prev, target, fade, status, phase: s.words[i] };
}

function drawArcs(E: Env, f: ArcsFrame) {
  const M = E.narrow ? 8 : 16, T = E.s.T;
  paintArcs(E, { x0: M, sp: (E.W - 2 * M) / T, yTok: E.H - (E.narrow ? 60 : 42), top: 16 }, f.q, f.prev, f.target, f.fade);
}

export const draw = drawArcs;
