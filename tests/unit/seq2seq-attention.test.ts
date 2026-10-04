import { describe, expect, test } from 'vitest';
import { createRng } from '../../src/lib/math/rng';
import {
  initParams, forward, backward, exp, tanh, train, translate, alignment, encode, decode, reverse, argmax, SIZE, H,
  encdecTiming, encdecFrame, attnTiming, attnFrame, bottleneckFrame, bottleneckT, LENS, alignFrame, alignT,
} from '../../src/components/widgets/Seq2SeqAttention/sim';

describe('seq2seq model', () => {
  test.each([false, true])('backward matches numerical gradients (attention %s)', (attn) => {
    const p = initParams(createRng(3)), g = new Float64Array(SIZE), x = [1, 5, 2, 7, 0, 3], y = reverse(x);
    backward(p, attn, x, y, g);
    for (let i = 0; i < SIZE; i += 7) {
      const o = p[i];
      p[i] = o + 1e-5; const a = forward(p, attn, x, y).loss;
      p[i] = o - 1e-5; const b = forward(p, attn, x, y).loss;
      p[i] = o;
      const num = (a - b) / 2e-5;
      expect(Math.abs(num - g[i])).toBeLessThan(1e-6 + 1e-5 * Math.abs(num));
    }
  });
  test('attention weights are a distribution and the context is their mix', () => {
    const p = initParams(createRng(5)), f = forward(p, true, [0, 1, 2, 3]);
    f.alpha.forEach((a, i) => {
      expect(a.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 12);
      for (let r = 0; r < H; r++) expect(f.ctx[i][r]).toBeCloseTo(a.reduce((s, v, j) => s + v * f.hs[j][r], 0), 12);
    });
    const q = forward(p, false, [0, 1, 2, 3]);
    q.ctx.forEach((c) => expect(c).toBe(q.hs[3])); // the fixed vector: always the last encoder state
  });
  test('exp and tanh built from basic operations match Math to rounding', () => {
    for (let x = -50; x <= 50; x += 0.0731) {
      expect(Math.abs(exp(x) / Math.exp(x) - 1)).toBeLessThan(1e-13);
      expect(Math.abs(tanh(x) - Math.tanh(x))).toBeLessThan(1e-15);
    }
  });
  test('letters round-trip', () => {
    expect(decode(encode('hdfab'))).toBe('hdfab');
    expect(encode('abz')).toEqual([0, 1]);
  });
});

describe('training (the lesson claims)', () => {
  const plain = train(false), attn = train(true);
  test('the fixed vector fails on long strings, attention does not', () => {
    expect(plain.seqAcc[0]).toBeGreaterThan(0.95); // length 3
    expect(plain.seqAcc[LENS.length - 1]).toBeLessThan(0.1); // length 10
    for (const a of attn.seqAcc) expect(a).toBeGreaterThan(0.9);
  });
  test('the learned alignment is the mirror image', () => {
    const x = encode('cegbhadf'), a = alignment(attn.params, x);
    a.alpha.forEach((row, i) => expect(argmax(row)).toBe(x.length - 1 - i));
    expect(decode(a.out)).toBe('fdahbgec');
    expect(decode(translate(attn.params, true, x).probs.map(argmax))).toBe('fdahbgec');
  });
  test('training is deterministic', () => {
    expect(train(false, 300).params).toEqual(train(false, 300).params);
  });
}, 120000);

describe('timelines', () => {
  test('encoder-decoder phases', () => {
    const tm = encdecTiming(5);
    expect(encdecFrame(0, tm).phase).toBe('encode');
    expect(encdecFrame(tm.encEnd + 0.1, tm).phase).toBe('hand over');
    const end = encdecFrame(tm.T, tm);
    expect(end.phase).toBe('decode'); end.enc.forEach((v) => expect(v).toBe(1)); end.dec.forEach((v) => expect(v).toBe(1));
  });
  test('attention steps through every output', () => {
    const tm = attnTiming(4);
    expect(attnFrame(0, tm).i).toBe(-1);
    const end = attnFrame(tm.T, tm);
    expect(end.i).toBe(3); end.done.forEach((v) => expect(v).toBe(1));
  });
  test('bottleneck and alignment endpoints', () => {
    expect(bottleneckFrame(0).L).toBe(LENS[0]);
    expect(bottleneckFrame(bottleneckT).L).toBe(LENS[LENS.length - 1]);
    expect(alignFrame(0, 31).k).toBe(0);
    expect(alignFrame(alignT(31), 31).k).toBe(30);
  });
});
