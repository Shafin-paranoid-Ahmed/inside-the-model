// Pure math for the landing-page loss landscape: surface, gradient, optimizers, mesh, ray-cast.
// No DOM, no WebGL.

export const EXTENT = 3; // surface spans [-EXTENT, EXTENT]²

type Bump = { a: number; cx: number; cy: number; s: number };
// Negative a = well, positive a = hill. Deep global well, shallower local well, a ridge between.
const BUMPS: Bump[] = [
  { a: -1.7, cx: 1.25, cy: -0.95, s: 0.75 },
  { a: -0.8, cx: -1.35, cy: 1.05, s: 0.4 },
  { a: 0.45, cx: -0.1, cy: 0.1, s: 0.45 },
  { a: 0.5, cx: 1.6, cy: 1.6, s: 0.6 },
];
const BOWL = 0.07, RIP = 0.1;

export function loss(x: number, y: number): number {
  let z = BOWL * (x * x + y * y) + RIP * Math.sin(1.7 * x) * Math.cos(1.5 * y);
  for (const b of BUMPS) {
    const dx = x - b.cx, dy = y - b.cy;
    z += b.a * Math.exp(-(dx * dx + dy * dy) / b.s);
  }
  return z;
}

export function grad(x: number, y: number): [number, number] {
  let gx = 2 * BOWL * x + RIP * 1.7 * Math.cos(1.7 * x) * Math.cos(1.5 * y);
  let gy = 2 * BOWL * y - RIP * 1.5 * Math.sin(1.7 * x) * Math.sin(1.5 * y);
  for (const b of BUMPS) {
    const dx = x - b.cx, dy = y - b.cy;
    const g = b.a * Math.exp(-(dx * dx + dy * dy) / b.s) * (-2 / b.s);
    gx += g * dx;
    gy += g * dy;
  }
  return [gx, gy];
}

// ---- optimizers ---------------------------------------------------------------------------

export type OptName = 'SGD' | 'Momentum' | 'Adam';
export const OPTS: OptName[] = ['SGD', 'Momentum', 'Adam'];

export type Ball = {
  name: OptName;
  x: number; y: number;
  m: [number, number]; // momentum / Adam first moment
  v: [number, number]; // Adam second moment
  t: number;
  trail: number[]; // flat x,y pairs, oldest first
  done: boolean;
};

export const TRAIL = 220;

export function makeBall(name: OptName, x: number, y: number): Ball {
  return { name, x, y, m: [0, 0], v: [0, 0], t: 0, trail: [x, y], done: false };
}

/** One optimizer step. Returns true once the ball has settled. */
export function stepBall(b: Ball): boolean {
  if (b.done) return true;
  const [gx, gy] = grad(b.x, b.y);
  let dx: number, dy: number;
  b.t++;
  if (b.name === 'SGD') {
    dx = -0.03 * gx; dy = -0.03 * gy;
  } else if (b.name === 'Momentum') {
    b.m[0] = 0.95 * b.m[0] + gx; b.m[1] = 0.95 * b.m[1] + gy;
    dx = -0.012 * b.m[0]; dy = -0.012 * b.m[1];
  } else {
    const b1 = 0.9, b2 = 0.999, lr = 0.035;
    b.m[0] = b1 * b.m[0] + (1 - b1) * gx; b.m[1] = b1 * b.m[1] + (1 - b1) * gy;
    b.v[0] = b2 * b.v[0] + (1 - b2) * gx * gx; b.v[1] = b2 * b.v[1] + (1 - b2) * gy * gy;
    const c1 = 1 - b1 ** b.t, c2 = 1 - b2 ** b.t;
    dx = (-lr * (b.m[0] / c1)) / (Math.sqrt(b.v[0] / c2) + 1e-8);
    dy = (-lr * (b.m[1] / c1)) / (Math.sqrt(b.v[1] / c2) + 1e-8);
  }
  b.x = clamp(b.x + dx, -EXTENT, EXTENT);
  b.y = clamp(b.y + dy, -EXTENT, EXTENT);
  b.trail.push(b.x, b.y);
  if (b.trail.length > 2 * TRAIL) b.trail.splice(0, 2);
  b.done = b.t > 900 || (Math.hypot(dx, dy) < 2e-4 && Math.hypot(gx, gy) < 0.02);
  return b.done;
}

const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));

/** Seeded start points on the local-minimum side, where optimizers disagree most. */
export function startPoint(k: number): [number, number] {
  const r = mulberry(k + 1);
  return [-2.3 + 0.8 * r(), -2.9 + 0.9 * r()];
}

function mulberry(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---- mesh ---------------------------------------------------------------------------------

/** Height scale from loss to world units. */
export const H = 0.85;
export const LOW = -1.75, HIGH = 1.25; // loss range used for tinting

/** Interleaved [x, y, z, nx, ny, nz, t] per vertex (y is up, t = normalized height) + triangle indices. */
export function buildMesh(n: number) {
  const verts = new Float32Array(n * n * 7);
  const idx = new Uint32Array((n - 1) * (n - 1) * 6);
  for (let j = 0; j < n; j++)
    for (let i = 0; i < n; i++) {
      const x = -EXTENT + (2 * EXTENT * i) / (n - 1), y = -EXTENT + (2 * EXTENT * j) / (n - 1);
      const z = loss(x, y);
      const [gx, gy] = grad(x, y);
      // normal of (x, H·f, y): (-H·fx, 1, -H·fy)
      const nx = -H * gx, nz = -H * gy, len = Math.hypot(nx, 1, nz);
      const o = 7 * (j * n + i);
      verts[o] = x; verts[o + 1] = H * z; verts[o + 2] = y;
      verts[o + 3] = nx / len; verts[o + 4] = 1 / len; verts[o + 5] = nz / len;
      verts[o + 6] = (z - LOW) / (HIGH - LOW);
    }
  let k = 0;
  for (let j = 0; j < n - 1; j++)
    for (let i = 0; i < n - 1; i++) {
      const a = j * n + i, b = a + 1, c = a + n, d = c + 1;
      idx[k++] = a; idx[k++] = c; idx[k++] = b;
      idx[k++] = b; idx[k++] = c; idx[k++] = d;
    }
  return { verts, idx };
}

// ---- camera + ray-cast --------------------------------------------------------------------

export type Vec3 = [number, number, number];
export const TARGET: Vec3 = [0, -0.15, 0];
export const DIST = 11.2, FOV = (32 * Math.PI) / 180;

export function eye(yaw: number, pitch: number): Vec3 {
  return [
    TARGET[0] + DIST * Math.cos(pitch) * Math.sin(yaw),
    TARGET[1] + DIST * Math.sin(pitch),
    TARGET[2] + DIST * Math.cos(pitch) * Math.cos(yaw),
  ];
}

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a: Vec3): Vec3 => { const l = Math.hypot(...a); return [a[0] / l, a[1] / l, a[2] / l]; };
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/** Camera basis: forward, right, up. */
export function basis(e: Vec3) {
  const f = norm(sub(TARGET, e));
  const r = norm(cross(f, [0, 1, 0]));
  const u = cross(r, f);
  return { f, r, u };
}

/** Column-major view-projection matrix. */
export function viewProj(yaw: number, pitch: number, aspect: number): Float32Array {
  const e = eye(yaw, pitch), { f, r, u } = basis(e);
  const t = 1 / Math.tan(FOV / 2), near = 0.5, far = 40;
  const A = (far + near) / (near - far), B = (2 * far * near) / (near - far);
  // view rows: r, u, -f ; projection applied on top
  const V = [r, u, [-f[0], -f[1], -f[2]] as Vec3];
  const tr = [-dot(r, e), -dot(u, e), dot(f, e)];
  const m = new Float32Array(16);
  const P = [t / aspect, t, A];
  for (let c = 0; c < 3; c++) {
    m[c * 4 + 0] = P[0] * V[0][c];
    m[c * 4 + 1] = P[1] * V[1][c];
    m[c * 4 + 2] = P[2] * V[2][c];
    m[c * 4 + 3] = -V[2][c];
  }
  m[12] = P[0] * tr[0];
  m[13] = P[1] * tr[1];
  m[14] = P[2] * tr[2] + B;
  m[15] = -tr[2];
  return m;
}

/** Project a world point to [0,1]² screen coords (y down). Returns null if behind the camera. */
export function project(m: Float32Array, p: Vec3): [number, number] | null {
  const x = m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12];
  const y = m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13];
  const w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
  if (w <= 0) return null;
  return [(x / w + 1) / 2, (1 - y / w) / 2];
}

/** Ray from the camera through NDC (nx, ny ∈ [-1,1]) against the height field. Returns surface (x, y) or null. */
export function rayCast(yaw: number, pitch: number, aspect: number, nx: number, ny: number): [number, number] | null {
  const e = eye(yaw, pitch), { f, r, u } = basis(e), t = Math.tan(FOV / 2);
  const d = norm([
    f[0] + r[0] * nx * t * aspect + u[0] * ny * t,
    f[1] + r[1] * nx * t * aspect + u[1] * ny * t,
    f[2] + r[2] * nx * t * aspect + u[2] * ny * t,
  ]);
  const above = (s: number) => {
    const px = e[0] + d[0] * s, py = e[1] + d[1] * s, pz = e[2] + d[2] * s;
    if (Math.abs(px) > EXTENT || Math.abs(pz) > EXTENT) return 1;
    return py - H * loss(px, pz);
  };
  let prev = 0;
  for (let s = 0.05; s < 30; s += 0.05) {
    if (above(s) < 0) {
      let lo = prev, hi = s;
      for (let k = 0; k < 24; k++) { const mid = (lo + hi) / 2; if (above(mid) < 0) hi = mid; else lo = mid; }
      return [e[0] + d[0] * hi, e[2] + d[2] * hi];
    }
    prev = s;
  }
  return null;
}
