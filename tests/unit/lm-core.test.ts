import { describe, expect, test } from 'vitest';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { createRng } from '../../src/lib/math/rng';
import { initWeights, forward, prefill, decodeStep, softmax, parseModel, type Config, type Manifest, type Weights } from '../../src/lib/transformer';
import { CHARSET, VOCAB, UNKNOWN, encode, decode } from '../../src/lib/lm/charset';
import { softmaxT, topK, topP, pickIndex, entropy, generate } from '../../src/lib/lm/sample';
import { macsForward, weightBytes, kvBytes, roofline, CONFIGS } from '../../src/lib/lm/cost';

const cfg = (extra: Partial<Config> = {}): Config => ({ vocab: VOCAB, d: 8, heads: 2, layers: 2, context: 12, ff: 16,
  positions: 'learned', causal: true, norm: 'pre', tied: true, ...extra });

/** Plain-loop causal forward (pre-LN, learned positions, tied) that counts every multiply-add it does. */
function countingForward(w: Weights, ids: number[]): { logits: number[]; macs: number } {
  const c = w.config, d = c.d, dk = d / c.heads, p = w.params, o = (n: string) => w.offsets[n].offset;
  let macs = 0;
  const ln = (x: number[], n: string) => {
    const m = x.reduce((s, v) => s + v, 0) / d, v = x.reduce((s, y) => s + (y - m) ** 2, 0) / d, inv = 1 / Math.sqrt(v + 1e-5);
    return x.map((y, j) => (y - m) * inv * p[o(`${n}.gain`) + j] + p[o(`${n}.bias`) + j]);
  };
  const lin = (x: number[], n: string) => {
    const t = w.offsets[`${n}.weight`];
    return Array.from({ length: t.rows }, (_, r) => {
      let s = p[o(`${n}.bias`) + r];
      for (let j = 0; j < t.cols; j++) { s += p[t.offset + r * t.cols + j] * x[j]; macs++; }
      return s;
    });
  };
  let xs = ids.map((id, t) => Array.from({ length: d }, (_, j) => p[o('embedding') + id * d + j] + p[o('position') + t * d + j]));
  for (let l = 0; l < c.layers; l++) {
    const h = xs.map((x) => ln(x, `${l}.ln1`)), q = h.map((x) => lin(x, `${l}.q`)), k = h.map((x) => lin(x, `${l}.k`)), v = h.map((x) => lin(x, `${l}.v`));
    const att = xs.map((_, t) => {
      const out = new Array(d).fill(0);
      for (let hh = 0; hh < c.heads; hh++) {
        const scores = Array.from({ length: t + 1 }, (_, s) => {
          let dot = 0; for (let j = 0; j < dk; j++) { dot += q[t][hh * dk + j] * k[s][hh * dk + j]; macs++; }
          return dot / Math.sqrt(dk);
        });
        const a = softmax(scores);
        for (let s = 0; s <= t; s++) for (let j = 0; j < dk; j++) { out[hh * dk + j] += a[s] * v[s][hh * dk + j]; macs++; }
      }
      return out;
    });
    xs = xs.map((x, t) => { const y = lin(att[t], `${l}.o`); return x.map((z, j) => z + y[j]); });
    xs = xs.map((x) => { const y = lin(lin(ln(x, `${l}.ln2`), `${l}.ff1`).map((z) => Math.max(0, z)), `${l}.ff2`); return x.map((z, j) => z + y[j]); });
  }
  const logits = xs.flatMap((x) => {
    const f = ln(x, 'final');
    return c.vocab.map((_, id) => { let s = p[o('logit.bias') + id]; for (let j = 0; j < d; j++) { s += f[j] * p[o('embedding') + id * d + j]; macs++; } return s; });
  });
  return { logits, macs };
}

describe('lm charset', () => {
  test('round-trips CHARSET text and maps everything else to one unknown id', () => {
    expect(VOCAB.length).toBe(38); expect(CHARSET.length).toBe(37); expect(UNKNOWN).toBe(37);
    const text = 'alice said: "why, it\'s a-q?!;\n';
    expect(decode(encode(text))).toBe(text);
    expect(encode('Alice')).toEqual(encode('alice'));
    expect(encode('é1(')).toEqual([UNKNOWN, UNKNOWN, UNKNOWN]);
    expect(decode([UNKNOWN])).toBe('�');
  });
});

describe('lm sampling', () => {
  const z = [2, 1, 0];
  test('temperature reshapes softmax; T = 0 is greedy; adding a constant changes nothing', () => {
    const p1 = softmaxT(z, 1), p05 = softmaxT(z, 0.5);
    [0.665241, 0.244728, 0.090031].forEach((v, i) => expect(p1[i]).toBeCloseTo(v, 6));
    [0.866813, 0.117310, 0.015876].forEach((v, i) => expect(p05[i]).toBeCloseTo(v, 6));
    expect([...softmaxT(z, 0)]).toEqual([1, 0, 0]);
    const shifted = softmaxT(z.map((x) => x + 10), 1);
    p1.forEach((v, i) => expect(shifted[i]).toBeCloseTo(v, 14));
    expect(entropy(softmaxT(z, 1e6))).toBeCloseTo(Math.log(3), 6);
    expect(entropy([0.5, 0.5])).toBeCloseTo(Math.LN2, 14);
  });
  test('top-k and top-p keep the right ids and renormalise', () => {
    const p = [0.05, 0.5, 0.1, 0.3, 0.05];
    const k = topK(p, 2);
    expect(k.ids).toEqual([1, 3]); expect(k.probs[1]).toBeCloseTo(0.625, 14); expect(k.probs[0]).toBe(0);
    expect(topP(p, 0.85).ids).toEqual([1, 3, 2]);
    expect(topP(p, 0.8).ids).toEqual([1, 3]); // reaching p exactly is enough
    expect(topP(p, 1).ids.length).toBe(5);
    expect(topP(p, 0.85).probs.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 14);
  });
  test('pickIndex is the inverse CDF and skips zero-probability ids', () => {
    const p = [0.2, 0, 0.5, 0.3];
    expect(pickIndex(p, 0)).toBe(0); expect(pickIndex(p, 0.19)).toBe(0); expect(pickIndex(p, 0.2)).toBe(2);
    expect(pickIndex(p, 0.699)).toBe(2); expect(pickIndex(p, 0.7)).toBe(3); expect(pickIndex(p, 0.9999999)).toBe(3);
  });
  test('generation: greedy ignores us, same us is reproducible, cache and sliding window match full forwards', () => {
    const w = initWeights(cfg(), createRng(3)), r = createRng(4), us = Array.from({ length: 30 }, () => r.next()), prompt = encode('the ');
    const a = generate(w, prompt, 30, { temperature: 0.8, topP: 0.9 }, us), b = generate(w, prompt, 30, { temperature: 0.8, topP: 0.9 }, us);
    expect(a.ids).toEqual(b.ids);
    expect(generate(w, prompt, 20, { temperature: 0 }, []).ids).toEqual(generate(w, prompt, 20, { temperature: 0 }, us).ids);
    // Every step's model distribution equals a full forward over the last `context` tokens.
    a.steps.forEach((s, i) => {
      const ids = a.ids.slice(0, prompt.length + i).slice(-12), f = forward(w, ids).probs.slice(-VOCAB.length);
      for (let j = 0; j < VOCAB.length; j++) expect(Math.abs(s.model[j] - f[j])).toBeLessThan(1e-12);
      expect(s.logp).toBeCloseTo(Math.log(s.model[s.id]), 14);
    });
    expect(a.ids.length).toBe(34);
    expect(() => generate(w, prompt, 5, { temperature: 1 }, [0.1])).toThrow();
  });
});

describe('lm cost', () => {
  test('macsForward equals an op-counting forward pass with identical logits', () => {
    for (const extra of [{}, { d: 12, heads: 3, ff: 20, layers: 1 }]) {
      const w = initWeights(cfg(extra), createRng(5)), ids = encode('alice was'), counted = countingForward(w, ids);
      const core = forward(w, ids).logits;
      counted.logits.forEach((v, i) => expect(Math.abs(v - core[i])).toBeLessThan(1e-10));
      expect(macsForward(w.config, ids.length, ids.length)).toBe(counted.macs);
      // A cached decode of the last token costs the difference between the two full passes.
      expect(macsForward(w.config, 1, ids.length)).toBe(counted.macs - countingForward(w, ids.slice(0, -1)).macs);
    }
  });
  test('prefill plus one-token decode steps does exactly the work of one full pass', () => {
    const c = cfg();
    let sum = 0; for (let t = 1; t <= 12; t++) sum += macsForward(c, 1, t);
    expect(sum).toBe(macsForward(c, 12, 12));
    expect(macsForward(c, 5, 12) + macsForward(c, 7, 7)).toBe(macsForward(c, 12, 12));
  });
  test('memory, weights and roofline from published configs', () => {
    const l2 = CONFIGS.llama2_7b, l3 = CONFIGS.llama3_8b, gpu = CONFIGS.h100_sxm;
    expect(kvBytes({ layers: l2.layers, kvDim: l2.kvDim, tokens: 1, bytes: 2 })).toBe(512 * 1024);
    expect(kvBytes({ layers: l2.layers, kvDim: l2.kvDim, tokens: 4096, bytes: 2 })).toBe(2 * 1024 ** 3);
    expect(kvBytes({ layers: l3.layers, kvDim: l3.kvDim, tokens: 8192, bytes: 2 })).toBe(1024 ** 3);
    expect(weightBytes(cfg(), 1)).toBe(initWeights(cfg(), createRng(1)).params.length);
    const r = roofline(gpu, 1);
    expect(r.bound).toBe('memory'); expect(r.flops).toBe(3.35e12); expect(r.ridge).toBeCloseTo(295.37, 2);
    expect(roofline(gpu, 1000)).toMatchObject({ flops: 989.5e12, bound: 'compute' });
    for (const entry of Object.values(CONFIGS)) expect(entry.source).toMatch(/^https:\/\//);
  });
});

describe('shipped Day 10 models and logs', () => {
  const ship = (name: string) => {
    const manifest: Manifest = JSON.parse(readFileSync(`public/models/${name}.json`, 'utf8')), bytes = new Uint8Array(readFileSync(`public/models/${name}.bin`));
    expect(createHash('sha256').update(bytes).digest('hex')).toBe(manifest.sha256);
    return { manifest, bytes, w: parseModel(manifest, bytes) };
  };
  test('icl: small int8 model infers the shift from the examples in the default prompt', () => {
    const { manifest, bytes, w } = ship('icl');
    expect(bytes.length).toBeLessThanOrEqual(15 * 1024);
    const ids = [...'c>f g>b a>d e>h g>'].map((c) => manifest.vocab.indexOf(c)), p = forward(w, ids).probs.slice(-manifest.vocab.length);
    expect(p[manifest.vocab.indexOf('b')]).toBeGreaterThan(0.9);
    const shots = manifest.metrics.shots as { k: number; accuracy: number }[];
    expect(shots[0].accuracy).toBeLessThan(0.25); expect(shots[4].accuracy).toBeGreaterThan(0.8);
    expect(readFileSync('public/models/icl-log.json').length).toBeLessThanOrEqual(15 * 1024);
  });
  test('facts: accuracy rises with training frequency, recomputed from the shipped bytes', () => {
    const { manifest, bytes, w } = ship('facts');
    expect(bytes.length).toBeLessThanOrEqual(40960);
    const data = JSON.parse(readFileSync('public/models/facts-data.json', 'utf8')), V = manifest.vocab.length;
    const acc = (count: number) => {
      const people = data.people.filter((p: { count: number }) => p.count === count);
      return people.filter((p: { name: string; month: number }) => {
        const row = forward(w, [...`${p.name} was born in `].map((c) => manifest.vocab.indexOf(c))).probs.slice(-V).slice(28, 40);
        return row.indexOf(Math.max(...row)) === p.month;
      }).length / people.length;
    };
    const by = manifest.metrics.byCount as { count: number; accuracy: number }[];
    for (const count of [20, 1]) expect(acc(count)).toBeCloseTo(by.find((b) => b.count === count)!.accuracy, 10);
    expect(acc(20)).toBeGreaterThan(acc(1));
  });
  test('scaling sweep and house log respect their budgets', () => {
    const sweep = readFileSync('public/models/scaling-sweep.json'), runs = JSON.parse(sweep.toString()).runs;
    expect(sweep.length).toBeLessThanOrEqual(15 * 1024); expect(runs.length).toBe(42);
    for (const r of runs) expect(r.C).toBe(6 * r.N * r.D);
    expect(readFileSync('public/models/house-log.json').length).toBeLessThanOrEqual(6 * 1024);
  });
});

describe('shipped house model', () => {
  const manifest: Manifest = JSON.parse(readFileSync('public/models/house.json', 'utf8'));
  const bytes = new Uint8Array(readFileSync('public/models/house.bin'));
  const w = parseModel(manifest, bytes);
  test('checksum, size and config', () => {
    expect(createHash('sha256').update(bytes).digest('hex')).toBe(manifest.sha256);
    expect(bytes.length).toBeLessThanOrEqual(40960);
    expect(manifest.dtype).toBe('int8');
    expect(manifest.vocab).toEqual(VOCAB);
  });
  test('quantised validation loss reproduces the manifest and beats the bigram baseline', () => {
    const ids = encode(readFileSync('scripts/data/alice.txt', 'utf8')), size = Math.floor(ids.length / 50);
    const valid = [9, 19, 29, 39, 49].map((i) => ids.slice(i * size, i === 49 ? ids.length : (i + 1) * size));
    let loss = 0, n = 0;
    for (const c of valid) for (let k = 0; k < Math.floor((c.length - 1) / 64); k++) {
      const win = c.slice(k * 64, k * 64 + 65), f = forward(w, win.slice(0, -1));
      for (let t = 0; t < 64; t++) { loss -= Math.log(f.probs[t * VOCAB.length + win[t + 1]]); n++; }
    }
    const m = manifest.metrics as Record<string, number>;
    expect(loss / n).toBeCloseTo(m.validLoss, 10);
    expect(m.validLoss).toBeLessThan(m.bigramValidCrossEntropy - 0.3);
  }, 60_000);
  test('KV-cache decode equals the full forward on the shipped weights', () => {
    const ids = encode('alice was beginning to get very tired of sitting by her sister on the bank'.slice(0, 64));
    const full = forward(w, ids).logits, pre = prefill(w, ids.slice(0, 1));
    for (let t = 1; t < ids.length; t++) {
      const step = decodeStep(w, pre.cache, ids[t]);
      for (let i = 0; i < VOCAB.length; i++) expect(Math.abs(step[i] - full[t * VOCAB.length + i])).toBeLessThan(1e-12);
    }
  });
});
