import { describe, expect, test, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { createRng } from '../../src/lib/math/rng';
import { exp, tanh } from '../../src/lib/transformer/math';
import { exp as referenceExp, tanh as referenceTanh } from '../../src/components/widgets/Seq2SeqAttention/sim';
import { evaluate } from '../../src/lib/transformer/report';
import {
  TINY_CONFIG, initWeights, forward, prefill, decodeStep, generate, parameterLayout,
  lossAndGrad, makeAdam, adamStep, allowedNext, sampleSentence, encodeWords, VOCAB, SAMPLE_200,
  toFloat16, fromFloat16, encodeWeights, parseModel, loadModel, type Config, type Manifest,
} from '../../src/lib/transformer';

const small = (extra: Partial<Config> = {}) => initWeights({ ...TINY_CONFIG, d: 4, heads: 2, layers: 1, ff: 5, context: 6, ...extra }, createRng(12));
const closeArrays = (a: ArrayLike<number>, b: ArrayLike<number>, tolerance = 1e-10) => {
  expect(a.length).toBe(b.length);
  for (let i = 0; i < a.length; i++) expect(Math.abs(a[i] - b[i])).toBeLessThan(tolerance);
};

describe('transformer core', () => {
  test('arithmetic functions match the established deterministic implementation', () => {
    for (const x of [-700, -23, -2.1, 0, 0.7, 20, 700]) {
      expect(exp(x)).toBe(referenceExp(x)); expect(tanh(x)).toBe(referenceTanh(x));
    }
  });
  test('flat offsets cover every parameter exactly once; seeds reproduce initialization', () => {
    const a = initWeights(TINY_CONFIG, createRng(2)), b = initWeights(TINY_CONFIG, createRng(2));
    expect(a.params).toEqual(b.params);
    let end = 0;
    for (const t of Object.values(a.offsets)) { expect(t.offset).toBe(end); end += t.length; }
    expect(end).toBe(a.params.length); expect(end).toBe(7330);
    expect(() => parameterLayout({ ...TINY_CONFIG, heads: 3 })).toThrow();
  });
  test('the grammar accepts seeded samples and enforces number and object agreement', () => {
    const rng = createRng(72);
    for (let n = 0; n < 300; n++) {
      const ids = sampleSentence(rng);
      expect(ids.length).toBeLessThanOrEqual(10);
      for (let i = 0; i < ids.length; i++) expect(allowedNext(ids.slice(0, i)).has(ids[i])).toBe(true);
      expect(allowedNext(ids).size).toBe(0);
    }
    expect(allowedNext(encodeWords('[start] these small sheep')).has(VOCAB.indexOf('chase'))).toBe(true);
    expect(allowedNext(encodeWords('[start] these small sheep')).has(VOCAB.indexOf('chases'))).toBe(false);
    const food = allowedNext(encodeWords('[start] this cat eats the'));
    expect(food.has(VOCAB.indexOf('fish'))).toBe(true); expect(food.has(VOCAB.indexOf('dog'))).toBe(false);
    expect(() => allowedNext(encodeWords('[start] this cats'))).toThrow();
    expect(() => encodeWords('unknown')).toThrow('Unknown word');
  });
  for (const norm of ['pre', 'post'] as const) for (const positions of ['learned', 'rope'] as const) {
    test(`all manual parameter gradients match finite differences (${norm}, ${positions})`, () => {
      const w = small({ norm, positions }), example = { ids: [0, 3, 12], targets: [3, 12, 26] };
      const analytic = lossAndGrad(w, [example]);
      const eps = 1e-5;
      for (let i = 0; i < w.params.length; i++) {
        const original = w.params[i];
        w.params[i] = original + eps; const up = lossAndGrad(w, [example]).loss;
        w.params[i] = original - eps; const down = lossAndGrad(w, [example]).loss;
        w.params[i] = original;
        expect(Math.abs(analytic.grad[i] - (up - down) / (2 * eps)), `parameter ${i}`).toBeLessThan(2e-5);
      }
    });
    test(`cached prefill/decode equals full forward at every position (${norm}, ${positions})`, () => {
      const w = small({ norm, positions, layers: 2 }), ids = [0, 3, 8, 12, 26, 5];
      const opts = { ropeScale: (p: number) => p / 2, positionScale: 0.7 };
      const first = prefill(w, ids.slice(0, 2), opts);
      closeArrays(first.logits, forward(w, ids.slice(0, 2), opts).logits);
      for (let i = 2; i < ids.length; i++) {
        const next = decodeStep(w, first.cache, ids[i]);
        closeArrays(next, forward(w, ids.slice(0, i + 1), opts).logits.slice(-VOCAB.length));
        expect(first.cache.length).toBe(i + 1);
      }
      expect(() => decodeStep(w, first.cache, 1)).toThrow('context');
    });
  }
  test('editing a future token leaves all earlier causal outputs bit-identical', () => {
    const w = small({ layers: 2 }), before = forward(w, [0, 2, 8, 12, 26]), after = forward(w, [0, 2, 8, 14, 26]);
    expect(before.logits.slice(0, 3 * VOCAB.length)).toEqual(after.logits.slice(0, 3 * VOCAB.length));
    for (const l of before.layers) for (const a of l.weights) {
      for (let t = 0; t < 5; t++) {
        expect(a.subarray(t * 5, (t + 1) * 5).reduce((s, p) => s + p, 0)).toBeCloseTo(1, 12);
        for (let s = t + 1; s < 5; s++) expect(a[t * 5 + s]).toBe(0);
      }
    }
  });
  test('unmasked attention without positions is permutation equivariant', () => {
    const w = small({ layers: 2 }), ids = [0, 2, 8, 12], permutation = [2, 0, 3, 1];
    const a = forward(w, ids, { causal: false, noPositions: true });
    const b = forward(w, permutation.map((i) => ids[i]), { causal: false, noPositions: true });
    for (let t = 0; t < 4; t++) closeArrays(b.logits.slice(t * VOCAB.length, (t + 1) * VOCAB.length), a.logits.slice(permutation[t] * VOCAB.length, (permutation[t] + 1) * VOCAB.length));
  });
  test('ablations zero a head and patches replace only the named stream position', () => {
    const w = small({ layers: 2 }), ids = [0, 2, 12];
    const ablated = forward(w, ids, { ablate: [[0, 1]] });
    expect(ablated.layers[0].headOut[1].every((x) => x === 0)).toBe(true);
    const replacement = Float64Array.from([1, 2, 3, 4]);
    const base = forward(w, ids), patched = forward(w, ids, { patch: [{ layer: 0, position: 1, stream: 'resid2', value: replacement }] });
    expect(patched.layers[0].resid2.slice(4, 8)).toEqual(replacement);
    expect(patched.layers[0].resid2.slice(0, 4)).toEqual(base.layers[0].resid2.slice(0, 4));
  });
  test('attention-only untied models and soft-target losses have correct gradients', () => {
    const w = small({ ff: 0, tied: false }), ids = [0, 2, 12];
    const p = new Float64Array(VOCAB.length); p[3] = 0.4; p[5] = 0.6;
    const ex = { ids, targets: [-1, 3, -1], targetProbs: [p, p, p] };
    const a = lossAndGrad(w, [ex], { noPositions: true, noScale: true, causal: false });
    for (const tensor of Object.values(w.offsets)) {
      const i = tensor.offset, v = w.params[i], eps = 1e-5;
      w.params[i] = v + eps; const up = lossAndGrad(w, [ex], { noPositions: true, noScale: true, causal: false }).loss;
      w.params[i] = v - eps; const down = lossAndGrad(w, [ex], { noPositions: true, noScale: true, causal: false }).loss;
      w.params[i] = v;
      expect(a.grad[i]).toBeCloseTo((up - down) / (2 * eps), 5);
    }
  });
  test('an Adam update reduces a repeated-example loss and is deterministic', () => {
    const a = small(), b = small(), ex = [{ ids: [0, 3, 12], targets: [3, 12, 26] }];
    const sa = makeAdam(a.params.length), sb = makeAdam(b.params.length), initial = lossAndGrad(a, ex).loss;
    for (let i = 0; i < 8; i++) { adamStep(a.params, lossAndGrad(a, ex).grad, sa, 0.01); adamStep(b.params, lossAndGrad(b, ex).grad, sb, 0.01); }
    expect(lossAndGrad(a, ex).loss).toBeLessThan(initial); expect(a.params).toEqual(b.params);
  });
  test('generation honors a logit mask and a caller-supplied deterministic sampler', () => {
    const w = initWeights(TINY_CONFIG, createRng(8));
    const out = generate(w, [0], { logitMask: allowedNext, stopId: 1, sampler: (p) => p.findIndex((v) => v > 0) });
    expect(allowedNext(out).size).toBe(0);
    const unmasked = generate(w, [0], { forward: { causal: false }, logitMask: allowedNext, stopId: 1, sampler: (p) => p.findIndex((v) => v > 0) });
    expect(unmasked).toEqual(out);
    expect(() => prefill(w, [0], { causal: false })).toThrow('causal');
    expect(() => forward(w, [])).toThrow();
  });
});

describe('model storage', () => {
  test('binary16 handles known encodings, subnormals, signed zero and ties-to-even', () => {
    const fixtures = [[0, 0], [-0, 0x8000], [1, 0x3c00], [-2, 0xc000], [65504, 0x7bff], [2 ** -24, 1], [2 ** -14, 0x400]];
    for (const [value, bits] of fixtures) { expect(toFloat16(value)).toBe(bits); expect(Object.is(fromFloat16(bits), value)).toBe(true); }
    expect(toFloat16(1 + 2 ** -11)).toBe(0x3c00);
    expect(toFloat16(1 + 2 ** -11 + 2 ** -40)).toBe(0x3c01);
    expect(toFloat16(1 + 3 * 2 ** -11)).toBe(0x3c02);
    expect(fromFloat16(toFloat16(Infinity))).toBe(Infinity);
    expect(Number.isNaN(fromFloat16(toFloat16(NaN)))).toBe(true);
    for (let bits = 0; bits < 65536; bits++) if ((bits & 0x7c00) !== 0x7c00) expect(toFloat16(fromFloat16(bits))).toBe(bits);
  });
  for (const dtype of ['float16', 'int8'] as const) test(`${dtype} round-trip uses tensor rows and validates the manifest`, () => {
    const w = small(), packed = encodeWeights(w, dtype);
    const manifest: Manifest = { version: 1, name: 'test', config: w.config, vocab: w.config.vocab, offsets: w.offsets,
      dtype, parameterCount: w.params.length, scales: packed.scales, metrics: {}, lossCurve: [], training: {} };
    const restored = parseModel(manifest, packed.bytes);
    expect(restored.params.length).toBe(w.params.length);
    for (const [name, tensor] of Object.entries(w.offsets)) for (let r = 0; r < tensor.rows; r++) for (let j = 0; j < tensor.cols; j++) {
      const i = tensor.offset + r * tensor.cols + j;
      const tolerance = dtype === 'int8' ? packed.scales![name][r] / 2 + 1e-12 : Math.abs(w.params[i]) * 0.0005 + 3e-8;
      expect(Math.abs(restored.params[i] - w.params[i])).toBeLessThanOrEqual(tolerance);
    }
    expect(() => parseModel(manifest, packed.bytes.subarray(1))).toThrow('length');
  });
  test('visible islands share a single lazy model fetch', async () => {
    const w = small(), packed = encodeWeights(w);
    const manifest: Manifest = { version: 1, name: 'unit-model', config: w.config, vocab: w.config.vocab, offsets: w.offsets,
      dtype: 'float16', parameterCount: w.params.length, metrics: {}, lossCurve: [], training: {} };
    const mock = vi.fn(async (url: string) => url.endsWith('.json') ? new Response(JSON.stringify(manifest)) : new Response(packed.bytes.slice().buffer));
    vi.stubGlobal('fetch', mock);
    try { const a = loadModel('unit-model'), b = loadModel('unit-model'); expect(a).toBe(b); await a; expect(mock).toHaveBeenCalledTimes(2); }
    finally { vi.unstubAllGlobals(); }
  });
  // Assets must exist: acceptance cannot quietly skip untrained or missing models.
  for (const name of ['tiny-lm', 'tiny-lm-nomask']) test(`shipped ${name} reproduces its rounded-weight metrics`, () => {
    const manifest: Manifest = JSON.parse(readFileSync(`public/models/${name}.json`, 'utf8'));
    const bytes = new Uint8Array(readFileSync(`public/models/${name}.bin`));
    expect(createHash('sha256').update(bytes).digest('hex')).toBe(manifest.sha256);
    const w = parseModel(manifest, bytes);
    let total = 0, count = 0;
    for (const ids of SAMPLE_200) {
      const input = ids.slice(0, -1), f = forward(w, input);
      for (let t = 0; t < input.length; t++) { for (const id of allowedNext(input.slice(0, t + 1))) total += f.probs[t * VOCAB.length + id]; count++; }
    }
    const mass = total / count;
    expect(mass).toBeCloseTo(manifest.metrics.grammaticalMass as number, 12);
    const report = evaluate(w);
    expect(report.heads).toEqual(manifest.metrics.heads);
    expect(report.distinctHeads).toEqual(manifest.metrics.distinctHeads);
    if (name === 'tiny-lm') {
      expect(mass).toBeGreaterThanOrEqual(0.97);
      expect(report.distinctHeads.pair.length).toBe(2);
      expect(report.distinctHeads.pair[0][0]).toBe(report.distinctHeads.pair[1][0]);
      expect(report.distinctHeads.meanTotalVariation).toBeGreaterThanOrEqual(0.2);
    }
  });
  test('eight embedding snapshots decode at the recorded offsets and finish at the shipped table', () => {
    const meta = JSON.parse(readFileSync('public/models/tiny-lm-snapshots.json', 'utf8'));
    const bytes = new Uint8Array(readFileSync('public/models/tiny-lm-snapshots.bin'));
    const manifest: Manifest = JSON.parse(readFileSync('public/models/tiny-lm.json', 'utf8'));
    const w = parseModel(manifest, new Uint8Array(readFileSync('public/models/tiny-lm.bin')));
    expect(createHash('sha256').update(bytes).digest('hex')).toBe(meta.sha256);
    expect(meta.checkpoints.length).toBe(8);
    expect(meta.checkpoints[0].step).toBe(0);
    expect(meta.checkpoints.at(-1).step).toBe(manifest.training.steps);
    const view = new DataView(bytes.buffer), size = w.offsets.embedding.length, start = meta.checkpoints.at(-1).byteOffset;
    expect(bytes.length).toBe(size * 2 * 8);
    for (let j = 0; j < size; j++) expect(fromFloat16(view.getUint16(start + j * 2, true))).toBe(w.params[w.offsets.embedding.offset + j]);
  });
});
