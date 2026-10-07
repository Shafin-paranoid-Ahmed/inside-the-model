import { describe, expect, it } from 'vitest';
import type { Frame, Kit, ModeImpl, Opts } from '../../src/components/widgets/_rl/kit';
import type { Env } from '../../src/components/widgets/_rl/svg';
import revise from '../../src/components/widgets/CritiqueRevise/m-revise';
import sl from '../../src/components/widgets/CritiqueRevise/m-sl';
import edit from '../../src/components/widgets/CritiqueRevise/m-edit';
import { trainAlignPPO } from '../../src/lib/align';

// Node stand-in for the page kit: memo works and worker jobs run inline.
const nodeKit = (): Kit => {
  const m = new Map<string, unknown>();
  return {
    memo: <T,>(k: string, f: () => T) => (m.has(k) ? m.get(k) : (m.set(k, f()), m.get(k))) as T,
    job: (fn: string, ...args: unknown[]) => (fn === 'trainAlignPPO' ? Promise.resolve(trainAlignPPO(args[0] as never)) : Promise.reject(new Error(fn))) as never,
  };
};
const ON = { answer: true, accurate: true, flattery: true, refuse: true, cautious: false };
const MODES: [string, ModeImpl<unknown, Frame>, Record<string, unknown>][] = [
  ['revise', revise as ModeImpl<unknown, Frame>, {}],
  ['sl', sl as ModeImpl<unknown, Frame>, {}],
  ['edit', edit as ModeImpl<unknown, Frame>, ON],
  ['edit', edit as ModeImpl<unknown, Frame>, { ...ON, cautious: true }],
];

describe.each(MODES)('mode %s %o', (name, mode, props) => {
  const o = { mode: name, ...props } as Opts, ui = { pick: null };
  it('markers increase, frames are deterministic at the endpoints and draw with text >= 11 px', async () => {
    const [a, b] = [await mode.setup(o, nodeKit()), await mode.setup(o, nodeKit())];
    const { duration, markers } = mode.timeline(a, o);
    markers.forEach((m, i) => { expect(m.at).toBeLessThan(duration); if (i) expect(m.at).toBeGreaterThan(markers[i - 1].at); });
    for (const t of [0, duration * 0.25, duration / 2, duration * 0.8, duration]) {
      const f = mode.frameAt(t, a, o, ui);
      expect(f.status.length).toBeGreaterThan(0);
      expect(mode.frameAt(t, b, o, ui)).toEqual(f);
      for (const narrow of [false, true]) {
        const E: Env = { W: narrow ? 360 : 720, H: narrow ? 560 : 400, narrow, out: [] };
        mode.draw(E, f, a, o);
        expect(E.out.length).toBeGreaterThan(0);
        expect(E.out.length).toBeLessThan(500);
        for (const sh of E.out) if (sh.k === 'text') expect(sh.size).toBeGreaterThanOrEqual(11);
      }
    }
    expect(mode.label(a, o).length).toBeGreaterThan(20);
  });
});

const end = async (mode: ModeImpl<unknown, Frame>, props: Record<string, unknown>) => {
  const o = { mode: 'x', ...props } as Opts, s = await mode.setup(o, nodeKit());
  return mode.frameAt(mode.timeline(s, o).duration, s, o, { pick: null });
};
const start = async (mode: ModeImpl<unknown, Frame>, props: Record<string, unknown>) => {
  const o = { mode: 'x', ...props } as Opts, s = await mode.setup(o, nodeKit());
  return mode.frameAt(0, s, o, { pick: null });
};

describe('numbers quoted in the lesson', () => {
  it('revise: 1.16 -> 0.77 principles broken per response, 206 of 256 rounds find nothing (starts from the base model)', async () => {
    const d = (await end(revise as ModeImpl<unknown, Frame>, { rounds: 4, seed: 3 })).detail!;
    expect(d).toContain('1.16 principles broken per response before, 0.77 after 4 rounds');
    expect(d).toContain('206 of 256 rounds');
  });
  it('revise: three starts (list, refusal, padded), each ends clean; the first chain is nothing, terse wrong, terse correct, nothing', async () => {
    const s = await revise.setup({ mode: 'revise' } as Opts, nodeKit());
    expect(s.shown.map((x) => x.y0)).toEqual([0, 6, 3]);
    expect(s.shown.map((x) => x.rounds[x.rounds.length - 1].after)).toEqual([0, 0, 0]);
    expect(s.shown[0].rounds.map((q) => (q.violated ? q.to : '-'))).toEqual(['-', 5, 1, '-']);
  });
  it('sl: violations 0.207 -> 0.110, gold 2.27 -> 2.51, terse wrong 0.14 -> 0.09; 12 of 64 changed', async () => {
    const a = await start(sl as ModeImpl<unknown, Frame>, {}), b = await end(sl as ModeImpl<unknown, Frame>, {});
    expect(a.readouts!.map(([, v]) => v)).toEqual(['0 of 150', '0.207', '2.27', '0.14']);
    expect(b.readouts!.map(([, v]) => v)).toEqual(['150 of 150', '0.110', '2.51', '0.09']);
    expect(a.detail).toContain('12 broke a principle, and 7 still do');
  });
  it('edit: default ends at hedge+refuse 0.358, gold 2.61; cautious principle gives 0.999 and 2.17', async () => {
    const d = await end(edit as ModeImpl<unknown, Frame>, ON), c = await end(edit as ModeImpl<unknown, Frame>, { ...ON, cautious: true });
    expect(d.readouts!.slice(1, 3).map(([, v]) => v)).toEqual(['0.358', '2.61']);
    expect(c.readouts!.slice(1, 3).map(([, v]) => v)).toEqual(['0.999', '2.17']);
    expect((await start(edit as ModeImpl<unknown, Frame>, ON)).readouts![1][1]).toBe('0.217');
  });
});
