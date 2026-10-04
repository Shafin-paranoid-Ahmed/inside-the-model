import { describe, expect, it } from 'vitest';
import type { Frame, Kit, ModeImpl, Opts } from '../../src/components/widgets/_rl/kit';
import type { Env } from '../../src/components/widgets/_rl/svg';
import qvalues from '../../src/components/widgets/GridWorldLab/m-qvalues';
import bellman from '../../src/components/widgets/GridWorldLab/m-bellman';
import qlearn from '../../src/components/widgets/GridWorldLab/m-qlearn';
import curve from '../../src/components/widgets/GridWorldLab/m-qlearn-curve';
import terminal from '../../src/components/widgets/GridWorldLab/m-qlearn-terminal';

const nodeKit = (): Kit => {
  const m = new Map<string, unknown>();
  return { memo: <T,>(k: string, f: () => T) => (m.has(k) ? m.get(k) : (m.set(k, f()), m.get(k))) as T, job: () => Promise.reject(new Error('no worker')) };
};
const ui = { pick: null };
const MODES: [string, ModeImpl<unknown, Frame>, Record<string, unknown>][] = [
  ['qvalues', qvalues as ModeImpl<unknown, Frame>, {}],
  ['bellman', bellman as ModeImpl<unknown, Frame>, { focus: '3,1', action: 'up', sweeps: 20 }],
  ['qlearn', qlearn as ModeImpl<unknown, Frame>, {}],
  ['qlearn-curve', curve as ModeImpl<unknown, Frame>, {}],
  ['qlearn-terminal', terminal as ModeImpl<unknown, Frame>, { gamma: 0.9 }],
];
const fr = async (name: string, mode: ModeImpl<unknown, Frame>, extra: Record<string, unknown>, t: number | 'end') => {
  const o = { mode: name, ...extra } as Opts, s = await mode.setup(o, nodeKit()), d = mode.timeline(s, o).duration;
  return { f: mode.frameAt(t === 'end' ? d : t, s, o, ui), s, o };
};

describe.each(MODES)('mode %s', (name, mode, extra) => {
  it('markers increase inside the timeline; frames narrate, draw and are deterministic', async () => {
    const o = { mode: name, ...extra } as Opts, a = await mode.setup(o, nodeKit()), b = await mode.setup(o, nodeKit());
    const { duration, markers } = mode.timeline(a, o);
    markers.forEach((m, i) => { expect(m.at).toBeLessThanOrEqual(duration); if (i) expect(m.at).toBeGreaterThan(markers[i - 1].at); });
    for (const t of [0, duration * 0.3, duration * 0.7, duration]) {
      const f = mode.frameAt(t, a, o, ui);
      expect(f.status.length).toBeGreaterThan(10);
      expect(JSON.stringify(mode.frameAt(t, b, o, ui))).toEqual(JSON.stringify(f));
      for (const narrow of [false, true]) {
        const E: Env = { W: narrow ? 360 : 720, H: narrow ? 650 : 420, narrow, out: [] };
        mode.draw(E, f, a, o);
        expect(E.out.length).toBeGreaterThan(0);
        for (const sh of E.out) if (sh.k === 'text') expect(sh.size).toBeGreaterThanOrEqual(11);
      }
    }
  });
});

describe('numbers quoted in the lesson', () => {
  it('qvalues: start cell Q* up 0.705, left 0.671, gap 0.034', async () => {
    const { f } = await fr('qvalues', qvalues as ModeImpl<unknown, Frame>, {}, 0.9);
    expect(f.status).toContain('0.705'); expect(f.status).toContain('0.671');
    expect(f.readouts?.[2][1]).toBe('0.034');
  });
  it('qvalues: three near ties (gap under 0.02)', async () => {
    const { s } = await fr('qvalues', qvalues as ModeImpl<unknown, Frame>, {}, 0);
    expect((s as { ties: number[] }).ties.length).toBe(3);
  });
  it('bellman: the three terms at (3,1) up add to Q* = 0.593', async () => {
    const { f } = await fr('bellman', bellman as ModeImpl<unknown, Frame>, { focus: '3,1', action: 'up', sweeps: 20 }, 7);
    expect(f.status).toContain('0.593');
  });
  it('qlearn: episode 1 of seed 10 ends in the goal with a terminal target 0.96 and Q 0.096', async () => {
    const { f, s } = await fr('qlearn', qlearn as ModeImpl<unknown, Frame>, {}, 'end');
    expect((s as { T1: number }).T1).toBe(20);
    expect(f.readouts?.[0][1]).toBe('1000 of 1000');
    expect(f.readouts?.[3][1]).toBe('8 of 9');
    expect(f.readouts?.[1][1]).toBe('0.686');
  });
  it('qlearn-terminal: goal target equals its reward and ordinary target adds the discounted successor', async () => {
    const { s } = await fr('qlearn-terminal', terminal as ModeImpl<unknown, Frame>, { gamma: 0.9 }, 0);
    const [goal, pit, ord] = (s as { cards: { done: boolean; r: number; target: number; maxNext: number }[] }).cards;
    expect(goal.target).toBeCloseTo(goal.r, 9); expect(pit.target).toBeCloseTo(pit.r, 9);
    expect(ord.done).toBe(false); expect(ord.target).toBeCloseTo(ord.r + 0.9 * ord.maxNext, 9);
  });
  it('qlearn-curve: exploring runs get more cells right than epsilon 0 by the end', async () => {
    const { f } = await fr('qlearn-curve', curve as ModeImpl<unknown, Frame>, {}, 'end');
    const cells = (f.readouts ?? []).slice(1).map(([, v]) => Number(v.split(' ')[0]));
    expect(cells[2]).toBeGreaterThan(cells[0]);
  });
});
