import { describe, expect, it } from 'vitest';
import { conv2d, map3 } from '../../src/lib/cnn/conv';
import { patches, patchWeights, project, shapeImage, embedPatches, poolPosition, initialize, objective, train, similarities, assemble, frameAt, probabilities } from '../../src/components/widgets/ClipLab/sim';

describe('ClipLab mathematics', () => {
  it('extracts every source pixel once in raster patch order', () => {
    const image = Float64Array.from({ length: 256 }, (_, i) => i);
    for (const p of [2, 4, 8]) {
      const result = patches(image, p);
      expect(result.length).toBe((16 / p) ** 2);
      expect(result.flatMap(x => [...x]).sort((a, b) => a - b)).toEqual([...image]);
      expect(result[1][0]).toBe(p);
    }
    expect(() => patches(image, 3)).toThrow();
  });
  it('shared patch projection equals kernel=stride convolution', () => {
    for (const p of [2, 4, 8]) {
      const image = shapeImage('triangle'), w = patchWeights(p);
      const conv = conv2d(map3(1, 16, 16, image), { inC: 1, outC: 8, k: p, stride: p, pad: 0, weight: w, bias: new Float64Array(8) });
      patches(image, p).forEach((x, i) => project(x, w).forEach((v, d) => expect(v).toBeCloseTo(conv.data[d * conv.h * conv.w + i], 12)));
    }
  });
  it('mean pooling ignores permutation, position then nonlinear mixing can distinguish it', () => {
    const tokens = embedPatches(shapeImage()), shuffled = [...tokens.slice(5), ...tokens.slice(0, 5)];
    poolPosition(tokens, false).forEach((v, i) => expect(v).toBeCloseTo(poolPosition(shuffled, false)[i], 12));
    expect(poolPosition(tokens, true)).not.toEqual(poolPosition(shuffled, true));
  });
  it('checks gradients through both normalization paths and learned temperature', () => {
    const model = initialize(), analytic = objective(model, true).g, h = 1e-5;
    for (const key of ['image', 'text'] as const) for (const i of [0, 21, 50, model[key].length - 1]) {
      const before = model[key][i]; model[key][i] = before + h; const plus = objective(model).loss;
      model[key][i] = before - h; const minus = objective(model).loss; model[key][i] = before;
      expect(analytic[key][i]).toBeCloseTo((plus - minus) / (2 * h), 5);
    }
    const before = model.logScale; model.logScale = before + h; const plus = objective(model).loss;
    model.logScale = before - h; const minus = objective(model).loss;
    expect(analytic.logScale).toBeCloseTo((plus - minus) / (2 * h), 5);
  });
  it('trains real matching probabilities and reports separate held-out performance', () => {
    const run = train(), first = run.snapshots[0], last = run.snapshots.at(-1)!;
    expect(first.loss).toBeCloseTo(2.119206, 5);
    expect(last.loss).toBeLessThan(0.001);
    expect(last.probability).toBeGreaterThan(0.999);
    expect(run.accuracy).toEqual([10 / 16, 9 / 16, 10 / 16]);
    const scores = similarities(run.model, shapeImage(), ['circle left of square', 'square left of circle', 'large circle', 'small circle']);
    expect(scores[0]).toBe(scores[1]); expect(scores[2] - scores[3]).toBeGreaterThan(0.7);
    expect(frameAt(0, 'contrastive', run).snapshot).toBe(first);
    expect(frameAt(10, 'contrastive', run).snapshot).toBe(last);
  });
  it('is deterministic and has normalized probabilities', () => {
    expect(train(10)).toEqual(train(10));
    expect(probabilities([1000, 1001, 999]).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
  });
  it('actually projects each image token and counts context tokens', () => {
    const output = assemble(shapeImage());
    expect(output.projected).toHaveLength(16); expect(output.projected[0]).toHaveLength(12);
    expect(output.count).toBe(23); expect(output.share).toBe(16 / 23);
    expect(assemble(shapeImage(), 2).count).toBe(71);
    expect(assemble(shapeImage(), 8).count).toBe(11);
  });
});
