import { describe, expect, it } from 'vitest';
import {
  loss, grad, makeBall, stepBall, OPTS, startPoint, buildMesh, viewProj, project, rayCast, H, LOW, HIGH,
} from '../../src/components/widgets/LossLandscape3D/sim';

describe('loss landscape', () => {
  it('analytic gradient matches central differences', () => {
    const h = 1e-5;
    for (const [x, y] of [[0, 0], [1.2, -0.9], [-1.3, 1], [2.1, 0.4], [-0.5, -2.2]]) {
      const [gx, gy] = grad(x, y);
      expect(gx).toBeCloseTo((loss(x + h, y) - loss(x - h, y)) / (2 * h), 6);
      expect(gy).toBeCloseTo((loss(x, y + h) - loss(x, y - h)) / (2 * h), 6);
    }
  });

  it('has a deeper global well than the local one, within the tint range', () => {
    const g = loss(1.25, -0.95), l = loss(-1.35, 1.05);
    expect(g).toBeLessThan(l);
    expect(g).toBeGreaterThan(LOW);
    expect(loss(-0.1, 0.1)).toBeLessThan(HIGH);
  });

  it('every optimizer settles at a point with small gradient', () => {
    for (const name of OPTS) {
      const [x, y] = startPoint(3);
      const b = makeBall(name, x, y);
      while (!stepBall(b));
      expect(Math.hypot(...grad(b.x, b.y))).toBeLessThan(0.05);
    }
  });

  it('optimizers disagree on some start: SGD stays local while Adam or Momentum reaches the global well', () => {
    let found = false;
    for (let k = 0; k < 20 && !found; k++) {
      const [x, y] = startPoint(k);
      const end = OPTS.map((n) => { const b = makeBall(n, x, y); while (!stepBall(b)); return loss(b.x, b.y); });
      if (end[0] > -1.2 && Math.min(end[1], end[2]) < -1.4) found = true;
    }
    expect(found).toBe(true);
  });

  it('mesh has unit normals and matching heights', () => {
    const n = 16, { verts, idx } = buildMesh(n);
    expect(idx.length).toBe((n - 1) ** 2 * 6);
    for (let v = 0; v < n * n; v++) {
      const o = 7 * v;
      expect(Math.hypot(verts[o + 3], verts[o + 4], verts[o + 5])).toBeCloseTo(1, 6);
      expect(verts[o + 1]).toBeCloseTo(H * loss(verts[o], verts[o + 2]), 5);
    }
  });

  it('ray-cast hits the surface point under the cursor (the nearest one if another is occluded)', () => {
    const yaw = 0.7, pitch = 0.6, aspect = 1.3, m = viewProj(yaw, pitch, aspect);
    for (const [x, y] of [[0.5, -0.5], [-1.3, 1], [1.8, 1.2]]) {
      const s = project(m, [x, H * loss(x, y), y])!;
      const hit = rayCast(yaw, pitch, aspect, s[0] * 2 - 1, 1 - s[1] * 2)!;
      const back = project(m, [hit[0], H * loss(hit[0], hit[1]), hit[1]])!;
      expect(back[0]).toBeCloseTo(s[0], 3);
      expect(back[1]).toBeCloseTo(s[1], 3);
    }
    // an unoccluded point comes back exactly
    const s = project(m, [0.5, H * loss(0.5, -0.5), -0.5])!;
    const hit = rayCast(yaw, pitch, aspect, s[0] * 2 - 1, 1 - s[1] * 2)!;
    expect(hit[0]).toBeCloseTo(0.5, 2);
    expect(hit[1]).toBeCloseTo(-0.5, 2);
  });
});
