import { describe, expect, it } from 'vitest';
import { advance, markerAt, nextMarker } from '../../src/components/widgets/_shared/clock';

const s = (t: number, playing = true, speed = 1) => ({ t, playing, speed });

describe('advance', () => {
  it('does nothing while paused', () => expect(advance(s(1, false), 0.5, { duration: 8 })).toEqual(s(1, false)));
  it('scales dt by speed', () => expect(advance(s(1, true, 2), 0.25, { duration: 8 }).t).toBeCloseTo(1.5));
  it('wraps at duration and carries the overshoot', () => expect(advance(s(7.9), 0.3, { duration: 8 }).t).toBeCloseTo(0.2));
  it('loops inside a range', () => {
    const o = { duration: 8, loopFrom: 4, loopTo: 6 };
    expect(advance(s(5.9), 0.3, o).t).toBeCloseTo(4.2);
    expect(advance(s(5), 0.5, o).t).toBeCloseTo(5.5);
  });
  it('can hold past duration and stays in range for big dt', () => {
    expect(advance(s(8.5), 0.4, { duration: 8, loopFrom: 6, loopTo: 8.7 }).t).toBeCloseTo(6.2);
    const t = advance(s(0), 100, { duration: 8 }).t;
    expect(t).toBeGreaterThanOrEqual(0);
    expect(t).toBeLessThan(8);
  });
  it('does not mutate its input', () => {
    const a = s(1);
    advance(a, 1, { duration: 8 });
    expect(a.t).toBe(1);
  });
});

const M = [{ at: 0, label: 'a' }, { at: 2, label: 'b' }, { at: 5, label: 'c' }];
describe('markers', () => {
  it('nextMarker finds the next strictly-later marker', () => {
    expect(nextMarker(0, M)).toBe(2);
    expect(nextMarker(1.99, M)).toBe(5); // within eps of 2 counts as already there
    expect(nextMarker(2, M)).toBe(5);
    expect(nextMarker(5, M)).toBeUndefined();
  });
  it('markerAt is the last marker at or before t', () => {
    expect(markerAt(0, M)?.label).toBe('a');
    expect(markerAt(4.9, M)?.label).toBe('b');
    expect(markerAt(9, M)?.label).toBe('c');
    expect(markerAt(-1, M)).toBeUndefined();
  });
});
