/** Pure math for SlopeExplorer: secant to tangent on a curve, and partial derivatives to gradient on a bowl. No DOM. */

export type FnId = 'square' | 'exp' | 'sigmoid';
export interface Fn {
  id: FnId; name: string; expr: string;
  f: (x: number) => number; df: (x: number) => number;
  xd: [number, number]; yd: [number, number]; x0: number;
}
const sig = (x: number) => 1 / (1 + Math.exp(-x));
export const FNS: Record<FnId, Fn> = {
  square: { id: 'square', name: 'x²', expr: 'x²', f: (x) => x * x, df: (x) => 2 * x, xd: [-3, 3], yd: [-1, 9], x0: 1 },
  exp: { id: 'exp', name: 'eˣ', expr: 'eˣ', f: Math.exp, df: Math.exp, xd: [-3, 2.5], yd: [-1, 9], x0: 0.5 },
  sigmoid: { id: 'sigmoid', name: 'sigmoid', expr: 'σ(x)', f: sig, df: (x) => sig(x) * (1 - sig(x)), xd: [-6, 6], yd: [-0.15, 1.15], x0: 0.5 },
};

export const smooth = (u: number) => { const c = Math.min(1, Math.max(0, u)); return c * c * (3 - 2 * c); };

// ---------- secant mode ----------
export const H0 = 2, H_MIN = 0.001, SECANT_T = 8, SECANT_RUN = 7;
/** Step size at time t: shrinks geometrically from 2 to 0.001, then holds. */
export const hAt = (t: number) => H0 * Math.pow(H_MIN / H0, Math.min(1, Math.max(0, t / SECANT_RUN)));
export const timeForH = (h: number) => (SECANT_RUN * Math.log(h / H0)) / Math.log(H_MIN / H0);
export const secantMarkers = () => [2, 0.5, 0.1, 0.01, 0.001].map((h) => ({ at: timeForH(h), label: `h=${h}` }));

export interface SecantFrame { h: number; x0: number; x1: number; y0: number; y1: number; secant: number; tangent: number; gap: number }
export function secantAt(t: number, fn: Fn, x0: number): SecantFrame {
  const h = hAt(t), y0 = fn.f(x0), y1 = fn.f(x0 + h), secant = (y1 - y0) / h, tangent = fn.df(x0);
  return { h, x0, x1: x0 + h, y0, y1, secant, tangent, gap: Math.abs(secant - tangent) };
}
/** Secant slope for n log-spaced values of h, from H0 down to H_MIN (for the side plot). */
export function slopeCurve(fn: Fn, x0: number, n = 60): { h: number; s: number }[] {
  return Array.from({ length: n }, (_, i) => {
    const h = H0 * Math.pow(H_MIN / H0, i / (n - 1));
    return { h, s: (fn.f(x0 + h) - fn.f(x0)) / h };
  });
}

// ---------- gradient mode ----------
export const bowl = (x: number, y: number) => 0.5 * x * x + y * y;
export const bowlGrad = (x: number, y: number): [number, number] => [x, 2 * y];
export const BOWL_D: [number, number] = [-3, 3];
export const BOWL_LEVELS = [0.5, 1, 2, 3.5, 5, 7, 9, 12];
export const GRAD_START: [number, number] = [2, 1.5];
export const LR = 0.2, N_STEPS = 10, ARROW = 0.4;
export const GRAD_T = 6 + N_STEPS * 0.8;
export const gradMarkers = () => [{ at: 0, label: '∂f/∂x' }, { at: 2, label: '∂f/∂y' }, { at: 4, label: '∇f' }, { at: 6, label: 'downhill' }];

/** Positions of the ball: start plus N_STEPS gradient steps against the gradient. */
export function ballPath(start: [number, number], lr = LR, n = N_STEPS): [number, number][] {
  const p: [number, number][] = [[...start]];
  for (let i = 0; i < n; i++) { const [x, y] = p[i], [gx, gy] = bowlGrad(x, y); p.push([x - lr * gx, y - lr * gy]); }
  return p;
}

export interface GradFrame {
  px: number; py: number; gx: number; gy: number;
  ex: number; ey: number; slide: number; eg: number;   // arrow growth 0..1 (x, y, y-slide, combined)
  ball: { x: number; y: number; f: number; k: number; alpha: number };
}
export function gradAt(t: number, start: [number, number], path = ballPath(start)): GradFrame {
  const [px, py] = start, [gx, gy] = bowlGrad(px, py);
  const s = Math.min(N_STEPS, Math.max(0, (t - 6) / 0.8)), k = Math.min(N_STEPS - 1, Math.floor(s));
  const a = path[k], b = path[k + 1], u = s >= N_STEPS ? 1 : smooth(s - k);
  const x = a[0] + (b[0] - a[0]) * u, y = a[1] + (b[1] - a[1]) * u;
  return {
    px, py, gx, gy,
    ex: smooth(t / 2), ey: smooth((t - 2) / 2), slide: smooth((t - 4) / 1.2), eg: smooth((t - 4) / 1.5),
    ball: { x, y, f: bowl(x, y), k: Math.round(s), alpha: smooth((t - 5.5) / 0.5) },
  };
}
