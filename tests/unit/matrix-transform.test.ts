import { describe, expect, it } from 'vitest';
import * as S from '../../src/components/widgets/MatrixTransform/sim';

const close = (a: number[], b: number[]) => a.forEach((v, i) => expect(v).toBeCloseTo(b[i], 9));

describe('MatrixTransform math', () => {
  it('columns of A are the images of the basis vectors', () => {
    const A: S.Mat2 = [2, 3, 5, 7];
    close(S.apply(A, [1, 0]), [2, 5]);
    close(S.apply(A, [0, 1]), [3, 7]);
  });
  it('matrix-vector product is a stack of row dot products', () => {
    close(S.apply([1, 2, 3, 4], [5, 6]), [1 * 5 + 2 * 6, 3 * 5 + 4 * 6]);
  });
  it('(AB)x = A(Bx) but AB differs from BA', () => {
    const A: S.Mat2 = [1, 1, 0, 1], B: S.Mat2 = [0, -1, 1, 0], x: S.Vec2 = [1.5, -2];
    close(S.apply(S.mul2(A, B), x), S.apply(A, S.apply(B, x)));
    expect(S.mul2(A, B)).not.toEqual(S.mul2(B, A));
  });
  it('determinant multiplies and is zero for the collapse preset', () => {
    const A = S.PRESETS.shear.m, B = S.PRESETS.scale.m;
    expect(S.det(S.mul2(A, B))).toBeCloseTo(S.det(A) * S.det(B));
    expect(S.det(S.PRESETS.collapse.m)).toBeCloseTo(0);
    expect(S.det(S.PRESETS.flip.m)).toBeCloseTo(-1);
    expect(S.det(S.PRESETS.rotate.m)).toBeCloseTo(1);
  });
  it('matAt goes from the identity to the target, rotations keep det 1 on the way', () => {
    for (const p of Object.values(S.PRESETS)) {
      close(S.matAt(p.m, 0), S.I2);
      close(S.matAt(p.m, 1), p.m);
    }
    expect(S.det(S.matAt(S.PRESETS.rotate.m, 0.4))).toBeCloseTo(1);
  });
});

describe('frameAt', () => {
  const x: S.Vec2 = [1.5, 1];
  it('transform: identity at the start, target at the end, deterministic', () => {
    const A = S.PRESETS.shear.m;
    close(S.frameAt(0, 'transform', A, x).M, S.I2);
    close(S.frameAt(S.duration('transform'), 'transform', A, x).M, A);
    close(S.frameAt(S.duration('transform'), 'transform', A, x).y, S.apply(A, x));
    expect(S.frameAt(2.3, 'dot', A, x)).toEqual(S.frameAt(2.3, 'dot', A, x));
  });
  it('compose: B first then A ends at the matrix product AB, and the ghost appears', () => {
    const A = S.PRESETS.shear.m, B = S.PRESETS.rotate.m;
    close(S.frameAt(0, 'compose', A, x, B).M, S.I2);
    close(S.frameAt(S.COMPOSE.second, 'compose', A, x, B).M, B);
    const end = S.frameAt(S.duration('compose'), 'compose', A, x, B);
    close(end.M, S.mul2(A, B));
    close(end.y, S.apply(A, S.apply(B, x)));
    expect(end.ghost).toBe(1);
    expect(S.frameAt(0, 'compose', A, x, B).ghost).toBe(0);
  });
  it('unit square shrinks to zero area when the collapse preset lands', () => {
    const P = S.squareImage(S.PRESETS.collapse.m);
    expect(P[1][0] * P[3][1] - P[1][1] * P[3][0]).toBeCloseTo(0);
  });
});

describe('shapes', () => {
  const A = S.crop(S.makeGrid(11), 2, 3), B = S.crop(S.makeGrid(23), 3, 4);
  const d: S.Dims = { m: 2, n: 3, q: 3, p: 4 };
  it('grids are seeded and transposition round-trips', () => {
    expect(S.makeGrid(11)).toEqual(S.makeGrid(11));
    expect(S.transposeGrid(S.transposeGrid(S.makeGrid(5)))).toEqual(S.makeGrid(5));
  });
  it('sweep partial sums end at the matmul entries', () => {
    const C = S.matmul(A, B);
    expect(C.length).toBe(2);
    expect(C[0].length).toBe(4);
    for (let k = 0; k < 8; k++) {
      const f = S.shapesFrameAt((k + 0.999) * S.DWELL, A, B, d);
      expect([f.row, f.col]).toEqual([Math.floor(k / 4), k % 4]);
      expect(f.terms).toBe(3);
      expect(f.partial).toBe(C[f.row][f.col]);
    }
    expect(S.shapesFrameAt(0, A, B, d).terms).toBe(0);
  });
  it('mismatched inner dimensions are invalid', () => {
    expect(S.shapesValid({ m: 2, n: 3, q: 2, p: 4 })).toBe(false);
    expect(S.shapesFrameAt(1, A, B, { m: 2, n: 3, q: 2, p: 4 }).valid).toBe(false);
    expect(S.shapesDuration(d)).toBeCloseTo(8 * S.DWELL);
  });
});
