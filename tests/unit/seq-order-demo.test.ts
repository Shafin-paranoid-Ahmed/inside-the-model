import { describe, expect, test } from 'vitest';
import { makeDemo, frameAt, timing, words } from '../../src/components/widgets/SeqOrderDemo/sim';

describe('SeqOrderDemo', () => {
  const d = makeDemo('the dog chased the cat', 'the cat chased the dog');
  test('bag of words cannot tell the two orders apart, the RNN can', () => {
    expect(d.dBow).toBeLessThan(1e-12);
    expect(d.dRnn).toBeGreaterThan(0.5);
  });
  test('different words do change the bag of words', () => {
    expect(makeDemo('the dog chased the cat', 'the dog chased the man').dBow).toBeGreaterThan(0.1);
  });
  test('words() cleans and caps', () => {
    expect(words('  The  DOG bit ')).toEqual(['the', 'dog', 'bit']);
    expect(words('a b c d e f g h')).toHaveLength(6);
  });
  test('frameAt endpoints', () => {
    const tm = timing(5);
    expect(frameAt(0, 5).fill.every((v) => v === 0)).toBe(true);
    const e = frameAt(tm.T, 5);
    expect(e.fill.every((v) => v === 1)).toBe(true);
    expect(e.cmp).toBe(1);
  });
});
