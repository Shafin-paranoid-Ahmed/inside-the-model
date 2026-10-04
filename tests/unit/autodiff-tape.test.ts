import { describe, it, expect } from 'vitest';
import { buildTape, costs, frameAt, DEFAULTS, T_REVERSE, B1, REC } from '../../src/components/widgets/AutodiffTape/sim';

const loss = (p: typeof DEFAULTS) => (Math.tanh(p.w1 * p.x1 + p.w2 * p.x2 + p.b) - p.t) ** 2;

describe('AutodiffTape sim', () => {
  const tape = buildTape();
  it('tape values match a direct evaluation and the tape is topologically ordered', () => {
    expect(tape.rows[tape.root].value).toBeCloseTo(loss(DEFAULTS), 12);
    expect(tape.root).toBe(tape.N - 1);
    for (const r of tape.rows) for (const a of r.args) expect(a).toBeLessThan(r.i);
  });
  it('final reverse adjoints equal numerical gradients', () => {
    const fin = tape.adj[tape.N];
    const h = 1e-6;
    for (const k of ['w1', 'w2', 'b'] as const) {
      const r = tape.rows.find((x) => x.label === k)!;
      const num = (loss({ ...DEFAULTS, [k]: DEFAULTS[k] + h }) - loss({ ...DEFAULTS, [k]: DEFAULTS[k] - h })) / (2 * h);
      expect(fin[r.i]).toBeCloseTo(num, 6);
    }
  });
  it('forward-mode tangents agree with reverse-mode adjoints', () => {
    const fin = tape.adj[tape.N];
    for (const pr of tape.paramRows) expect(tape.tangents(pr)[tape.root]).toBeCloseTo(fin[pr], 12);
  });
  it('the reverse sweep starts from a single seed', () => {
    expect(tape.adj[0][tape.root]).toBe(1);
    expect(tape.adj[0].filter((v) => v !== 0)).toHaveLength(1);
  });
  it('cost model: tie at n=1, forward mode grows linearly, reverse is constant', () => {
    expect(costs(13, 1).forward).toBe(costs(13, 1).reverse);
    expect(costs(13, 1000).forward).toBe(13 * 1001);
    expect(costs(13, 1e6).reverse).toBe(26);
  });
  it('frameAt endpoints', () => {
    const f0 = frameAt(0, tape, 'reverse', []);
    expect(f0.show.every((s) => s === 0)).toBe(true);
    expect(f0.phase).toBe('record');
    const fe = frameAt(T_REVERSE, tape, 'reverse', []);
    expect(fe.phase).toBe('done');
    tape.paramRows.forEach((r) => expect(fe.adj[r]).toBeCloseTo(tape.adj[tape.N][r], 12));
    expect(frameAt(B1 - 1e-9, tape, 'reverse', []).phase).toBe('backward');
    expect(frameAt(REC + 0.1, tape, 'reverse', []).adj[tape.root]).toBe(1);
  });
});
