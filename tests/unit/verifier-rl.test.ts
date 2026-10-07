import { describe, expect, it } from 'vitest';
import * as align from '../../src/lib/align';
import type { Frame, Kit, ModeImpl, Opts } from '../../src/components/widgets/_rl/kit';
import type { Env } from '../../src/components/widgets/_rl/svg';
import sample from '../../src/components/widgets/VerifierRL/m-sample';
import grpo, { pushOf } from '../../src/components/widgets/VerifierRL/m-grpo';
import train from '../../src/components/widgets/VerifierRL/m-train';
import length from '../../src/components/widgets/VerifierRL/m-length';
import domains, { openChange } from '../../src/components/widgets/VerifierRL/m-domains';
import { attempts } from '../../src/components/widgets/VerifierRL/vr-common';

// Node stand-in for the page kit: memo works, jobs run in-process.
const kit = (): Kit => {
  const m = new Map<string, unknown>();
  return {
    memo: <T,>(k: string, f: () => T) => (m.has(k) ? m.get(k) : (m.set(k, f()), m.get(k))) as T,
    job: <T,>(fn: string, ...args: unknown[]) => Promise.resolve((align as unknown as Record<string, (...a: unknown[]) => T>)[fn](...args)),
  };
};
const shared = kit();
const MODES: [string, ModeImpl<unknown, Frame>, Record<string, unknown>][] = [
  ['sample', sample as ModeImpl<unknown, Frame>, { seed: 0 }],
  ['grpo', grpo as ModeImpl<unknown, Frame>, { seed: 0 }],
  ['train', train as ModeImpl<unknown, Frame>, { verifier: 'strict' }],
  ['train', train as ModeImpl<unknown, Frame>, { verifier: 'lenient' }],
  ['length', length as ModeImpl<unknown, Frame>, {}],
  ['domains', domains as ModeImpl<unknown, Frame>, {}],
];

describe.each(MODES)('rlvr mode %s %o', (name, mode, props) => {
  const o = { mode: name, ...props } as Opts, ui = { pick: null };
  it('markers increase inside the timeline; frames at the endpoints are deterministic and draw with text >= 11 px', async () => {
    const a = await mode.setup(o, shared), b = await mode.setup(o, kit());
    const { duration, markers } = mode.timeline(a, o);
    markers.forEach((m, i) => { expect(m.at).toBeLessThanOrEqual(duration); if (i) expect(m.at).toBeGreaterThan(markers[i - 1].at); });
    for (const t of [0, duration * 0.35, duration]) {
      const f = mode.frameAt(t, a, o, ui);
      expect(f.status.length).toBeGreaterThan(0);
      expect(mode.frameAt(t, b, o, ui)).toEqual(f);
      for (const narrow of [false, true]) {
        const E: Env = { W: narrow ? 360 : 720, H: narrow ? 700 : 440, narrow, out: [] };
        mode.draw(E, f, a, o);
        expect(E.out.length).toBeGreaterThan(5);
        expect(E.out.length).toBeLessThan(500);
        for (const sh of E.out) if (sh.k === 'text') expect(sh.size).toBeGreaterThanOrEqual(11);
      }
    }
  });
});

const end = async (mode: ModeImpl<unknown, Frame>, o: Opts) => { const s = await mode.setup(o, shared); return mode.frameAt(mode.timeline(s, o).duration, s, o, { pick: null }); };
const readout = (f: Frame, label: string) => f.readouts?.find((r) => r[0] === label)?.[1];

describe('numbers quoted in the lesson', () => {
  it('sample (seed 0): 13 of 48 attempts pass, 4 of 6 problems solved, and 317 + 833 passes only with all its working', async () => {
    const g = attempts({ mode: 'sample', seed: 0 } as Opts);
    expect(g.rows.map((r) => `${r.p!.a}+${r.p!.b}`)).toEqual(['3+1', '32+18', '317+833', '7770+5739', '8+4', '28+68']);
    expect(g.rows.reduce((s, r) => s + r.strict.reduce((x, y) => x + y, 0), 0)).toBe(13);
    expect(g.rows.filter((r) => r.strict.some(Boolean)).length).toBe(4);
    const three = g.rows[2], winners = [...three.actions].filter((_, i) => three.strict[i]);
    expect(winners.length).toBe(1);
    expect(align.kOf(winners[0])).toBe(3);
    const f = await end(sample as ModeImpl<unknown, Frame>, { mode: 'sample', seed: 0 } as Opts);
    expect(readout(f, 'Passed')).toBe('13');
    expect(readout(f, 'Problems solved at least once')).toBe('4 of 6');
  });
  it('grpo (seed 0): 3 rows with no signal, largest advantage 2.646, push +2.65 and -1.89 on 317 + 833', async () => {
    const f = await end(grpo as ModeImpl<unknown, Frame>, { mode: 'grpo', seed: 0 } as Opts);
    expect(readout(f, 'With no signal')).toBe('3');
    expect(readout(f, 'Largest advantage')).toBe('+2.65');
    const g = attempts({ mode: 'grpo', seed: 0 } as Opts), push = pushOf(g.rows[2]);
    expect(Math.max(...push)).toBeCloseTo(2.6458, 3);
    expect(Math.min(...push)).toBeCloseTo(-1.8898, 3);
    expect(push.reduce((a, b) => a + b, 0)).toBeCloseTo(0, 9);
  });
  it('train strict (seed 5): accuracy 0.54 / 0.34 / 0.20 / 0.09 -> 0.99 / 0.99 / 0.98 / 0.95 at 50 -> 1.00 at 150', () => {
    const r = align.trainGRPO({ verifier: 'strict', iters: 150, G: 8, problems: 8, seed: 5 }), at = (it: number) => [0, 1, 2, 3].map((d) => +r.strict[it * 4 + d].toFixed(2));
    expect(at(0)).toEqual([0.54, 0.34, 0.2, 0.09]);
    expect(at(50)).toEqual([0.99, 0.99, 0.98, 0.95]);
    expect(at(150)).toEqual([1, 1, 1, 1]);
    const w = (it: number) => [0, 1, 2, 3].map((d) => +r.words[it * 4 + d].toFixed(1));
    expect(w(0)).toEqual([5.7, 7.3, 8.1, 8.4]);
    expect(w(150)).toEqual([10, 18.8, 27.7, 37.2]);
  });
  it('train lenient (seed 5): guess lists take over small problems and strict accuracy falls there', () => {
    const r = align.trainGRPO({ verifier: 'lenient', iters: 150, G: 8, problems: 8, seed: 5 }), at = (a: Float64Array, it: number) => [0, 1, 2, 3].map((d) => +a[it * 4 + d].toFixed(2));
    expect(at(r.lenient, 150)).toEqual([1, 1, 1, 1]);
    expect(at(r.list, 150)).toEqual([0.56, 0.71, 0.02, 0.01]);
    expect(at(r.strict, 0).slice(0, 2)).toEqual([0.54, 0.34]);
    expect(at(r.strict, 150)).toEqual([0.44, 0.29, 0.98, 0.99]);
  });
  it('domains: pass@8 at the start 0.65, pass@1 at the end 1.00, open prompts unchanged', () => {
    const r = align.trainGRPO({ verifier: 'strict', iters: 150, G: 8, problems: 8, seed: 5, domains: true });
    const m = (a: Float64Array, k: number) => (a[k * 4] + a[k * 4 + 1] + a[k * 4 + 2] + a[k * 4 + 3]) / 4;
    expect(m(r.pass8, 0)).toBeCloseTo(0.65, 2);
    expect(m(r.strict, 0)).toBeCloseTo(0.29, 2);
    expect(m(r.strict, 150)).toBeGreaterThan(0.995);
    expect(openChange(r, 150)).toBe(0);
  });
});
