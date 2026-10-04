import { describe, expect, it } from 'vitest';
import type { Frame, Kit, ModeImpl, Opts } from '../../src/components/widgets/_rl/kit';
import type { Env } from '../../src/components/widgets/_rl/svg';
import loop from '../../src/components/widgets/GridWorldLab/m-loop';
import explore from '../../src/components/widgets/BanditLab/m-explore';
import clipPositive from '../../src/components/widgets/PPOLab/m-clip-positive';

// Node stand-in for the page kit: memo works, worker jobs are not available here.
const nodeKit = (): Kit => {
  const m = new Map<string, unknown>();
  return {
    memo: <T,>(k: string, f: () => T) => (m.has(k) ? m.get(k) : (m.set(k, f()), m.get(k))) as T,
    job: () => Promise.reject(new Error('no worker in tests')),
  };
};
const MODES: [string, ModeImpl<unknown, Frame>][] = [
  ['loop', loop as ModeImpl<unknown, Frame>],
  ['explore', explore as ModeImpl<unknown, Frame>],
  ['clip-positive', clipPositive as ModeImpl<unknown, Frame>],
];

describe.each(MODES)('reference mode %s', (name, mode) => {
  const o = { mode: name } as Opts;
  const ui = { pick: null };

  it('markers are strictly increasing inside the timeline', async () => {
    const s = await mode.setup(o, nodeKit());
    const { duration, markers } = mode.timeline(s, o);
    expect(markers.length).toBeGreaterThan(1);
    markers.forEach((m, i) => {
      expect(m.at).toBeGreaterThanOrEqual(0);
      expect(m.at).toBeLessThanOrEqual(duration);
      if (i) expect(m.at).toBeGreaterThan(markers[i - 1].at);
    });
  });

  it('frames at start, middle and end narrate and draw, deterministically', async () => {
    const [a, b] = [await mode.setup(o, nodeKit()), await mode.setup(o, nodeKit())];
    const { duration } = mode.timeline(a, o);
    for (const t of [0, duration / 2, duration]) {
      const f = mode.frameAt(t, a, o, ui);
      expect(f.status.length).toBeGreaterThan(0);
      expect(mode.frameAt(t, b, o, ui)).toEqual(f);
      for (const narrow of [false, true]) {
        const E: Env = { W: narrow ? 360 : 720, H: narrow ? 640 : 400, narrow, out: [] };
        mode.draw(E, f, a, o);
        expect(E.out.length).toBeGreaterThan(0);
        for (const sh of E.out) if (sh.k === 'text') expect(sh.size).toBeGreaterThanOrEqual(11);
      }
    }
    expect(mode.label(a, o).length).toBeGreaterThan(20);
  });
});

it('explore reports the plan §6 last-100-step best-arm rates (greedy 74, ε 0.1 89, UCB 91)', async () => {
  const o = { mode: 'explore' } as Opts, s = await explore.setup(o, nodeKit());
  const end = explore.frameAt(explore.timeline(s, o).duration, s, o, { pick: null });
  expect(end.readouts?.slice(1).map(([, v]) => v)).toEqual(['74%', '89%', '91%']);
  expect(end.status).toContain('13 of 50');
});
