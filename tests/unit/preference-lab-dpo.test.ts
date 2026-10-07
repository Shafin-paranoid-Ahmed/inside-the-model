import { describe, expect, it } from 'vitest';
import * as align from '../../src/lib/align';
import type { Frame, Kit, ModeImpl, Opts } from '../../src/components/widgets/_rl/kit';
import type { Env } from '../../src/components/widgets/_rl/svg';
import optimal from '../../src/components/widgets/DPOLab/m-optimal';
import pair from '../../src/components/widgets/DPOLab/m-dpo-pair';
import train from '../../src/components/widgets/DPOLab/m-dpo-train';
import implicit from '../../src/components/widgets/DPOLab/m-implicit';
import compare from '../../src/components/widgets/DPOLab/m-compare';
import { prefData } from '../../src/components/widgets/PreferenceLab/sim';

// Node stand-in for the page kit: jobs call the align library directly instead of the worker.
const nodeKit = (): Kit => {
  const m = new Map<string, unknown>();
  return {
    memo: <T,>(k: string, f: () => T) => (m.has(k) ? m.get(k) : (m.set(k, f()), m.get(k))) as T,
    job: (fn: string, ...a: unknown[]) => Promise.resolve((align as unknown as Record<string, (...x: unknown[]) => unknown>)[fn](...a)) as Promise<never>,
  };
};
const shared = nodeKit();
const MODES: [string, ModeImpl<unknown, Frame>][] = [['optimal', optimal], ['dpo-pair', pair], ['dpo-train', train], ['implicit', implicit], ['compare', compare]] as never;

describe.each(MODES)('%s', (name, mode) => {
  const o = { mode: name } as Opts, ui = { pick: null };
  it('timeline markers increase; frames at the ends are deterministic and draw with text >= 11 px', async () => {
    const [a, b] = [await mode.setup(o, shared), await mode.setup(o, nodeKit())], { duration, markers } = mode.timeline(a, o);
    markers.forEach((m, i) => { expect(m.at).toBeLessThanOrEqual(duration); if (i) expect(m.at).toBeGreaterThan(markers[i - 1].at); });
    for (const t of [0, duration / 2, duration]) {
      const f = mode.frameAt(t, a, o, ui);
      expect(mode.frameAt(t, b, o, ui)).toEqual(f);
      for (const narrow of [false, true]) {
        const E: Env = { W: narrow ? 360 : 720, H: narrow ? 700 : 420, narrow, out: [] };
        mode.draw(E, f, a, o);
        expect(E.out.length).toBeGreaterThan(5);
        for (const sh of E.out) if (sh.k === 'text') expect(sh.size).toBeGreaterThanOrEqual(11);
      }
    }
  });
});

const end = async (mode: ModeImpl<unknown, Frame>, o: Opts, at?: number) => { const s = await mode.setup(o, shared); return mode.frameAt(at ?? mode.timeline(s, o).duration, s, o, { pick: null }); };
const ro = (f: Frame, label: string) => f.readouts!.find((r) => r[0].startsWith(label))![1];

describe('numbers quoted in the lesson', () => {
  it('optimal: holds at beta 0.1 within 0.126 of PPO; beta 0.02 hands prompt 0 to the rambling answer', async () => {
    const at = await end(optimal as never, { mode: 'optimal' } as Opts, 5);
    expect(at.status).toContain('0.126');
    expect(ro(at, 'Gold')).toBe('2.81'); expect(ro(at, 'KL')).toBe('0.99');
    const e = await end(optimal as never, { mode: 'optimal' } as Opts);
    expect(ro(e, 'Gold')).toBe('−0.44'); expect(ro(e, 'Rambling')).toBe('1.00');
    const s = await end(optimal as never, { mode: 'optimal' } as Opts, 0);
    expect(ro(s, 'Gold')).toBe('2.69'); expect(ro(s, 'KL')).toBe('0.18');
  });
  it('dpo-pair: loss 0.693 -> 0.383, margin 0 -> 0.762, weight 0.500 -> 0.318', async () => {
    const a = await end(pair as never, { mode: 'dpo-pair' } as Opts, 0), b = await end(pair as never, { mode: 'dpo-pair' } as Opts);
    expect([ro(a, 'Loss'), ro(a, 'Margin'), ro(a, 'Gradient')]).toEqual(['0.693', '0.000', '0.500']);
    expect([ro(b, 'Loss'), ro(b, 'Margin'), ro(b, 'Gradient')]).toEqual(['0.383', '0.762', '0.318']);
    expect(b.status).toContain('+1.01'); expect(b.status).toContain('−6.60');
  });
  it('dpo-train: end state, the mid-run detour, held-out accuracy', async () => {
    const f = await end(train as never, { mode: 'dpo-train' } as Opts);
    expect([ro(f, 'Loss'), ro(f, 'Gold'), ro(f, 'KL'), ro(f, 'Held')]).toEqual(['0.487', '2.82', '0.97', '0.680']);
    expect(f.status).toContain('0.005');
    const d = prefData({ mode: 'x' } as Opts, shared, 4), run = align.trainDPO(d.pairs, d.sft, { beta: 0.1, steps: 300 });
    expect(run.gold[100]).toBeCloseTo(1.97, 2); expect(run.kl[100]).toBeCloseTo(3.55, 2);
    expect(Math.min(...run.gold.slice(20, 160))).toBeLessThan(2.0);
  });
  it('the first 200 DPO steps are the reward-model fit step for step (v = beta (theta - theta_ref))', () => {
    const d = prefData({ mode: 'x' } as Opts, shared, 4), run = align.trainDPO(d.pairs, d.sft, { beta: 0.1, steps: 200 });
    for (const k of [1, 10, 100, 200]) for (let j = 0; j < 6; j++) expect(0.1 * (run.thetas[k][j] - d.sft[j])).toBeCloseTo(d.rm.ws[k][j], 4);
  });
  it('implicit: correlation 0.999, refuses has the largest weight gap 0.56', async () => {
    const f = await end(implicit as never, { mode: 'implicit' } as Opts);
    expect(ro(f, 'Correlation')).toBe('0.999'); expect(ro(f, 'Largest')).toBe('refuses 0.56');
  });
  it('compare: final gold and KL, chosen log-probability falls', async () => {
    const f = await end(compare as never, { mode: 'compare' } as Opts);
    expect(ro(f, 'β 0.1: PPO')).toBe('2.81 / 0.80'); expect(ro(f, 'β 0.1: DPO')).toBe('2.82 / 0.97');
    expect(ro(f, 'β 0.02: PPO')).toBe('−0.44 / 13.01'); expect(ro(f, 'β 0.1: mean')).toBe('−3.86'); expect(ro(f, 'β 0.02: mean')).toBe('−31.17');
  });
  it('worked check: log-ratios +0.5 and -0.3 at beta 0.1 give margin 0.08 and loss 0.654', () => {
    const m = 0.1 * (0.5 + 0.3);
    expect(m).toBeCloseTo(0.08, 12); expect(Math.log1p(Math.exp(-m))).toBeCloseTo(0.654, 3);
  });
});
