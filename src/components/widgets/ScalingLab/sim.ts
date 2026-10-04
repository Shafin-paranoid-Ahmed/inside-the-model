// ScalingLab: all numbers for the scaling-laws lesson. No DOM, no fetch.
// Measured modes use public/models/scaling-sweep.json (42 tiny character-level runs); `published` and
// `overtrain` use the published Chinchilla fit and need no sweep.

export type Mode = 'sweep' | 'compute' | 'params-data' | 'isoflop' | 'published' | 'extrapolate' | 'overtrain';
export type PubFit = 'hoffmann' | 'besiroglu';

export interface Run { d: number; N: number; nonEmbedding: number; D: number; C: number; validLoss: number; curve: number[][] }
export interface Sweep { runs: Run[]; batchTokens: number; widths: number[]; fit?: any }
export interface Pt { x: number; y: number }

// ---------- formatting ----------
const SUP = '⁰¹²³⁴⁵⁶⁷⁸⁹';
const sup = (n: number) => `${n < 0 ? '⁻' : ''}${String(Math.abs(n)).split('').map((c) => SUP[+c]).join('')}`;
export const fmtInt = (n: number) => Math.round(n).toLocaleString('en-US');
/** 5.88e23 -> "5.9×10²³". */
export function sci(x: number, digits = 2): string {
  if (x === 0) return '0';
  const e = Math.floor(Math.log10(Math.abs(x))), m = x / 10 ** e;
  const ms = m.toPrecision(digits);
  return ms === '10' || ms.startsWith('10.') ? `1×10${sup(e + 1)}` : `${ms}×10${sup(e)}`;
}
/** 70e9 -> "70B", 1.4e12 -> "1.4T", 2614 -> "2.6k". */
export function compact(n: number, digits = 2): string {
  const a = Math.abs(n);
  const f = (v: number, s: string) => `${+v.toPrecision(v < 10 ? digits : 3)}${s}`;
  return a >= 1e12 ? f(n / 1e12, 'T') : a >= 1e9 ? f(n / 1e9, 'B') : a >= 1e6 ? f(n / 1e6, 'M') : a >= 1e3 ? f(n / 1e3, 'k') : f(n, '');
}
export const logspace = (a: number, b: number, n: number): number[] => Array.from({ length: n }, (_, i) => a * (b / a) ** (i / (n - 1)));
export const ease = (u: number) => { const c = Math.min(1, Math.max(0, u)); return c * c * (3 - 2 * c); };

// ---------- measured sweep: frontier and power-law fits ----------
/** Best final validation loss reached at or below each compute (lower envelope of the run endpoints). */
export function frontier(runs: Run[]): Pt[] {
  let best = Infinity;
  return [...runs].sort((a, b) => a.C - b.C || a.validLoss - b.validLoss).flatMap((r) => {
    if (r.validLoss >= best) return [];
    best = r.validLoss;
    return [{ x: r.C, y: r.validLoss }];
  });
}
export interface Fit { E: number; a: number; b: number; r2: number; rmse: number }
export const predict = (f: Pick<Fit, 'E' | 'a' | 'b'>, c: number) => f.E + f.a * c ** -f.b;
/** Fit L = E + a·C^-b for a given floor E by least squares on log(L-E) against log C. */
export function fitAt(points: Pt[], E: number): Fit {
  const xs = points.map((p) => Math.log(p.x)), ys = points.map((p) => Math.log(p.y - E));
  const n = xs.length, mx = xs.reduce((s, v) => s + v, 0) / n, my = ys.reduce((s, v) => s + v, 0) / n;
  const b = -xs.reduce((s, x, i) => s + (x - mx) * (ys[i] - my), 0) / xs.reduce((s, x) => s + (x - mx) ** 2, 0);
  const f = { E, a: Math.exp(my + b * mx), b, r2: 0, rmse: 0 };
  const mean = points.reduce((s, p) => s + p.y, 0) / n;
  const sse = points.reduce((s, p) => s + (predict(f, p.x) - p.y) ** 2, 0), tot = points.reduce((s, p) => s + (p.y - mean) ** 2, 0);
  return { ...f, r2: tot ? 1 - sse / tot : 1, rmse: Math.sqrt(sse / n) };
}
/** Grid of candidate floors: 0 up to just under the lowest loss. */
export const floorGrid = (points: Pt[], steps = 600): number[] => {
  const top = Math.min(...points.map((p) => p.y)) * 0.98;
  return Array.from({ length: steps + 1 }, (_, k) => (k / steps) * top);
};
/** Best power law with a floor (grid search over E, least squares inside) or without one. Error is measured in loss units. */
export function fitPower(points: Pt[], floor = true): Fit {
  if (points.length < 2) throw new Error('A fit needs at least two points.');
  if (!floor) return fitAt(points, 0);
  let best = fitAt(points, 0);
  for (const E of floorGrid(points)) {
    const f = fitAt(points, E);
    if (Number.isFinite(f.rmse) && f.rmse < best.rmse) best = f;
  }
  return best;
}

// ---------- IsoFLOP from measured runs ----------
export interface IsoPoint { N: number; D: number; loss: number; measured: boolean }
/** For each model size, the loss at D = C/6N: exact if that size has a run there, otherwise interpolated in log D between its two nearest budgets. Never extrapolates, so sizes that would need more than one epoch are absent. */
export function isoPoints(runs: Run[], C: number): IsoPoint[] {
  return [...new Set(runs.map((r) => r.N))].sort((a, b) => a - b).flatMap((N): IsoPoint[] => {
    const rs = runs.filter((r) => r.N === N).sort((a, b) => a.D - b.D), D = C / (6 * N);
    if (D < rs[0].D * (1 - 1e-9) || D > rs[rs.length - 1].D * (1 + 1e-9)) return [];
    const exact = rs.find((r) => Math.abs(r.D / D - 1) < 1e-6);
    if (exact) return [{ N, D, loss: exact.validLoss, measured: true }];
    const hi = rs.findIndex((r) => r.D > D), a = rs[hi - 1], b = rs[hi];
    const u = Math.log(D / a.D) / Math.log(b.D / a.D);
    return [{ N, D, loss: a.validLoss + (b.validLoss - a.validLoss) * u, measured: false }];
  });
}

// ---------- fitted L(N, D) surface on our sweep (Chinchilla approach 3 on tiny models) ----------
export interface Surface { E: number; A: number; B: number; alpha: number; beta: number; rmse: number }
export const surfaceLoss = (s: Pick<Surface, 'E' | 'A' | 'B' | 'alpha' | 'beta'>, N: number, D: number) => s.E + s.A / N ** s.alpha + s.B / D ** s.beta;
function surfaceSolve(runs: Run[], E: number, alpha: number, beta: number, tabs: { n: number[]; d: number[]; ni: number[]; di: number[] }): Surface | null {
  const xn = tabs.n.map((N) => N ** -alpha), yd = tabs.d.map((D) => D ** -beta);
  let sxx = 0, sxy = 0, syy = 0, sxl = 0, syl = 0;
  for (let i = 0; i < runs.length; i++) {
    const x = xn[tabs.ni[i]], y = yd[tabs.di[i]], l = runs[i].validLoss - E;
    sxx += x * x; sxy += x * y; syy += y * y; sxl += x * l; syl += y * l;
  }
  const det = sxx * syy - sxy * sxy;
  if (!(Math.abs(det) > 0)) return null;
  const A = (sxl * syy - syl * sxy) / det, B = (syl * sxx - sxl * sxy) / det;
  if (!(A > 0 && B > 0)) return null;
  let sse = 0;
  for (let i = 0; i < runs.length; i++) sse += (E + A * xn[tabs.ni[i]] + B * yd[tabs.di[i]] - runs[i].validLoss) ** 2;
  return { E, A, B, alpha, beta, rmse: Math.sqrt(sse / runs.length) };
}
/** Grid over (E, alpha, beta), refined three times, with A and B solved exactly by least squares each time. */
export function fitSurface(runs: Run[]): Surface {
  const n = [...new Set(runs.map((r) => r.N))], d = [...new Set(runs.map((r) => r.D))];
  const tabs = { n, d, ni: runs.map((r) => n.indexOf(r.N)), di: runs.map((r) => d.indexOf(r.D)) };
  let best: Surface | null = null, step = [0.1, 0.05, 0.05];
  let centre = [1, 0.5, 0.5], span = [1, 0.5, 0.5];
  for (let round = 0; round < 4; round++) {
    const lo = [Math.max(0, centre[0] - span[0]), Math.max(0.02, centre[1] - span[1]), Math.max(0.02, centre[2] - span[2])];
    for (let E = lo[0]; E <= centre[0] + span[0] + 1e-9; E += step[0])
      for (let a = lo[1]; a <= centre[1] + span[1] + 1e-9; a += step[1])
        for (let b = lo[2]; b <= centre[2] + span[2] + 1e-9; b += step[2]) {
          const s = surfaceSolve(runs, E, a, b, tabs);
          if (s && (!best || s.rmse < best.rmse)) best = s;
        }
    if (!best) throw new Error('No valid surface fit');
    centre = [best.E, best.alpha, best.beta]; span = step.map((s) => s * 1.5); step = step.map((s) => s / 3);
  }
  return best!;
}
/** Compute-optimal size for a surface: minimise L(N, C/6N). */
export function surfaceOptimum(s: Surface, C: number): { N: number; D: number; loss: number } {
  const N = ((s.alpha * s.A) / (s.beta * s.B) * (C / 6) ** s.beta) ** (1 / (s.alpha + s.beta)), D = C / (6 * N);
  return { N, D, loss: surfaceLoss(s, N, D) };
}
const surfaceCache = new WeakMap<Sweep, Surface>();
export function sharedSurface(data: Sweep): Surface {
  let s = surfaceCache.get(data);
  if (!s) { s = fitSurface(data.runs); surfaceCache.set(data, s); }
  return s;
}

// ---------- published fits (VERIFIED 2026-10-04 against the papers) ----------
export interface Published { label: string; short: string; E: number; A: number; B: number; alpha: number; beta: number; source: string }
export const PUBLISHED: Record<PubFit, Published> = {
  // Hoffmann et al. 2022, arXiv:2203.15556, appendix D.2 eq. (10).
  hoffmann: { label: 'Hoffmann et al. 2022 (published fit)', short: 'Hoffmann 2022', E: 1.69, A: 406.4, B: 410.7, alpha: 0.34, beta: 0.28, source: 'https://arxiv.org/abs/2203.15556' },
  // Besiroglu et al. 2024, arXiv:2404.10102, eq. (3): re-estimate from the data extracted from Hoffmann's figure.
  besiroglu: { label: 'Besiroglu et al. 2024 (published refit)', short: 'Besiroglu 2024', E: 1.8172, A: 482.01, B: 2085.43, alpha: 0.3478, beta: 0.3658, source: 'https://arxiv.org/abs/2404.10102' },
};
/** Published model sizes and token counts (Hoffmann et al. 2022; Gopher 280B on about 300B tokens, Chinchilla 70B on 1.4T tokens). */
export const GOPHER = { N: 280e9, D: 300e9, name: 'Gopher' }, CHINCHILLA = { N: 70e9, D: 1.4e12, name: 'Chinchilla' };
/** Chinchilla's own compute budget for 6ND with its published N and D. */
export const CHINCHILLA_C = 6 * CHINCHILLA.N * CHINCHILLA.D;
/** Kaplan et al. 2020, arXiv:2001.08361, Table 6: N_opt = 1.3e9 · C_min^0.73 (non-embedding parameters, C_min in PF-days; 1 PF-day = 8.64e19 FLOP). */
export const KAPLAN = { Ne: 1.3e9, pN: 0.73, pD: 0.27, pfDay: 8.64e19, loss: { alphaN: 0.076, alphaD: 0.095, alphaC: 0.050 } };

export const publishedLoss = (N: number, D: number, p: Published) => p.E + p.A / N ** p.alpha + p.B / D ** p.beta;
/** N that minimises L(N, C/6N): closed form from dL/dN = 0. */
export const optimalN = (C: number, p: Published) => ((p.alpha * p.A) / (p.beta * p.B) * (C / 6) ** p.beta) ** (1 / (p.alpha + p.beta));
export const kaplanN = (C: number) => KAPLAN.Ne * (C / KAPLAN.pfDay) ** KAPLAN.pN;
/** Exponent in N_opt ∝ C^a implied by a fit. */
export const allocExponent = (p: Published) => p.beta / (p.alpha + p.beta);
/** Smallest N that can reach `target` with unlimited data. */
export const minParams = (target: number, p: Published) => (p.A / (target - p.E)) ** (1 / p.alpha);
/** Tokens needed for a model of N parameters to reach `target`; Infinity if N is too small. */
export function tokensForLoss(N: number, target: number, p: Published): number {
  const rest = target - p.E - p.A / N ** p.alpha;
  return rest > 0 ? (p.B / rest) ** (1 / p.beta) : Infinity;
}
/** Total FLOP: training 6ND plus serving 2N per generated token. */
export const totalCost = (N: number, target: number, served: number, p: Published) => N * (6 * tokensForLoss(N, target, p) + 2 * served);
export function servingOptimum(served: number, target: number, p: Published): { N: number; D: number; ratio: number; cost: number } {
  const lo = minParams(target, p) * 1.0001, grid = logspace(lo, 1e13, 800);
  let k = 0;
  for (let i = 1; i < grid.length; i++) if (totalCost(grid[i], target, served, p) < totalCost(grid[k], target, served, p)) k = i;
  let a = Math.log(grid[Math.max(0, k - 1)]), b = Math.log(grid[Math.min(grid.length - 1, k + 1)]);
  for (let i = 0; i < 60; i++) {
    const l = a + (b - a) / 3, r = b - (b - a) / 3;
    if (totalCost(Math.exp(l), target, served, p) < totalCost(Math.exp(r), target, served, p)) b = r; else a = l;
  }
  const N = Math.exp((a + b) / 2), D = tokensForLoss(N, target, p);
  return { N, D, ratio: D / N, cost: totalCost(N, target, served, p) };
}

// ---------- scenes ----------
export interface Tick { v: number; label: string }
export interface Axis { min: number; max: number; log: boolean; label: string; ticks: Tick[] }
export interface Line { pts: Pt[]; color: string; w?: number; dash?: boolean; op?: number }
export interface Dot { x: number; y: number; color: string; r?: number; hollow?: boolean; ring?: boolean }
export interface Label { x: number; y: number; text: string; anchor?: 'start' | 'middle' | 'end'; dx?: number; dy?: number; color?: string }
export interface Frame { lines: Line[]; dots: Dot[]; labels: Label[]; phase: string; readout: string; detail: string }
export interface Scene { mode: Mode; x: Axis; y: Axis; duration: number; markers: { at: number; label: string }[]; label: string; frame: (t: number, hover?: number) => Frame; hoverables?: { x: number; y: number; id: number }[] }

export interface Settings { floor: boolean; swap: boolean; fitPoints: number; budget: number; pubFit: PubFit }
export const DEFAULTS: Settings = { floor: true, swap: false, fitPoints: 4, budget: 9.7, pubFit: 'besiroglu' };

const decades = (lo: number, hi: number, fmt: (v: number) => string): Tick[] => {
  const out: Tick[] = [];
  for (let e = Math.ceil(Math.log10(lo) - 1e-9); 10 ** e <= hi * (1 + 1e-9); e++) out.push({ v: 10 ** e, label: fmt(10 ** e) });
  return out;
};
const lossTicks = (lo: number, hi: number, step: number): Tick[] => {
  const out: Tick[] = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) out.push({ v: +v.toFixed(3), label: v.toFixed(step < 0.1 ? 2 : 1) });
  return out;
};
const flopTick = (v: number) => `10${sup(Math.round(Math.log10(v)))}`;
const cmix = (u: number) => `color-mix(in srgb, var(--viz-seq-3) ${Math.round(100 - u * 100)}%, var(--viz-seq-5) ${Math.round(u * 100)}%)`;
const phaseAt = (t: number, markers: { at: number; label: string }[]) => {
  let p = markers[0].label;
  for (const m of markers) if (t >= m.at) p = m.label;
  return p;
};
const flopAxis = (lo: number, hi: number, label = 'Training compute C = 6ND (FLOP, log scale)'): Axis => ({ min: lo, max: hi, log: true, label, ticks: decades(lo, hi, flopTick) });
const sizeAxis = (lo: number, hi: number): Axis => ({ min: lo, max: hi, log: true, label: 'Parameters N (log scale)', ticks: decades(lo, hi, (v) => compact(v)) });
const lossAxis = (lo: number, hi: number, step = 0.2): Axis => ({ min: lo, max: hi, log: false, label: 'Validation loss (nats per character)', ticks: lossTicks(lo, hi, step) });

/** The lower envelope of curve points: for the sweep scene. */
const sizeU = (data: Sweep, d: number) => data.widths.indexOf(d) / (data.widths.length - 1);

export function sweepScene(data: Sweep): Scene {
  const runs = data.runs, env = frontier(runs);
  const curves = runs.map((r) => {
    const pts = r.curve.slice(0, -1).map(([step, y]) => ({ x: 6 * r.N * data.batchTokens * step, y }));
    pts.push({ x: r.C, y: r.validLoss }); // the last point is the full-validation result
    return pts;
  });
  const markers = [{ at: 0, label: 'All runs train' }, { at: 8, label: 'Best so far' }];
  const x = flopAxis(3e6, 3e11), y = lossAxis(2, 4.2);
  const ids = curves.map((c, i) => c.map((p) => ({ x: p.x, y: p.y, id: i }))).flat();
  const nMin = Math.min(...runs.map((r) => r.N)), nMax = Math.max(...runs.map((r) => r.N));
  const lowest = Math.min(...runs.map((r) => r.validLoss));
  return {
    mode: 'sweep', x, y, duration: 12, markers, label: 'Loss against compute for 42 training runs of different sizes', hoverables: ids,
    frame(t, hover = -1) {
      const cut = t < 8 ? x.min * (x.max / x.min) ** ease(t / 8) : Infinity;
      const lines: Line[] = [], dots: Dot[] = [], labels: Label[] = [];
      let soFar = Infinity;
      const live = hover >= 0 && curves[hover][0].x <= cut ? hover : -1;
      curves.forEach((c, i) => {
        const pts: Pt[] = [];
        for (let k = 0; k < c.length; k++) {
          if (c[k].x <= cut) pts.push(c[k]);
          else {
            if (k > 0 && c[k - 1].x < cut) { const u = Math.log(cut / c[k - 1].x) / Math.log(c[k].x / c[k - 1].x); pts.push({ x: cut, y: c[k - 1].y + (c[k].y - c[k - 1].y) * u }); }
            break;
          }
        }
        for (const p of pts) soFar = Math.min(soFar, p.y);
        if (pts.length > 1) lines.push({ pts, color: cmix(sizeU(data, runs[i].d)), w: live === i ? 3 : 1.4, op: live >= 0 && live !== i ? 0.3 : 0.9 });
      });
      if (t >= 8) {
        const span = Math.log(env[env.length - 1].x / env[0].x);
        const shown = t >= 12 ? env.length : env.filter((p) => Math.log(p.x / env[0].x) / span <= (t - 8) / 4 + 1e-9).length;
        const pts = env.slice(0, shown);
        if (pts.length > 1) lines.push({ pts, color: 'var(--accent)', w: 2.5 });
        for (const p of pts) dots.push({ x: p.x, y: p.y, color: 'var(--accent)', r: 3.5 });
      }
      let readout = `${runs.length} runs: ${fmtInt(nMin)} to ${fmtInt(nMax)} parameters, each trained on 8,448 to 270,592 characters. ${t >= 8 ? 'Lowest loss reached' : 'Lowest loss so far'}: ${(t >= 8 ? lowest : soFar).toFixed(4)} nats per character.`;
      if (live >= 0) {
        const r = runs[live];
        readout = `Hovered run: ${fmtInt(r.N)} parameters, ${fmtInt(r.D)} training characters, ${sci(r.C)} FLOP. It ended at loss ${r.validLoss.toFixed(4)}.`;
      }
      return { lines, dots, labels, phase: phaseAt(t, markers), readout, detail: t >= 8 ? 'Accent dots mark the lowest loss reached at or below each compute.' : 'Each curve is one run, drawn as far as the compute axis has reached.' };
    },
  };
}

/** Candidate floors used by the animated search (real fits at each). */
export function searchPath(points: Pt[], k = 48): Fit[] {
  const best = fitPower(points, true);
  return Array.from({ length: k + 1 }, (_, i) => fitAt(points, (best.E * i) / k));
}

function fitScene(mode: 'compute' | 'extrapolate', data: Sweep, s: Settings): Scene {
  const env = frontier(data.runs), last = env[env.length - 1];
  const n = mode === 'compute' ? env.length : Math.max(3, Math.min(env.length - 3, Math.round(s.fitPoints)));
  const used = env.slice(0, n), held = env.slice(n);
  const full = fitPower(used, s.floor);
  const path = mode === 'compute' ? (s.floor ? searchPath(used) : [full]) : [full];
  const x0 = env[0].x * 0.7, x1 = last.x * 1.4;
  const markers = mode === 'compute'
    ? [{ at: 0, label: 'Search for the floor' }, { at: 8, label: 'Best fit' }]
    : [{ at: 0, label: 'Fit' }, { at: 3, label: 'Predict' }, { at: 6, label: 'Check' }];
  const duration = 12;
  const x = flopAxis(x0, x1);
  const y: Axis = mode === 'compute'
    ? { min: 1.1, max: 3.3, log: true, label: s.floor ? 'Loss above the floor, L − E (log scale)' : 'Loss L (log scale)', ticks: [1.2, 1.5, 2, 3].map((v) => ({ v, label: String(v) })) }
    : lossAxis(2, 3.2);
  return {
    mode, x, y, duration, markers, label: mode === 'compute' ? 'Best loss against compute with a fitted power law' : 'A power law fitted on the smallest budgets and extended to larger ones',
    frame(t) {
      const lines: Line[] = [], dots: Dot[] = [], labels: Label[] = [];
      if (mode === 'compute') {
        const k = s.floor ? Math.round(ease(t / 8) * (path.length - 1)) : 0, f = path[k];
        const Y = (p: Pt) => p.y - f.E;
        for (const p of used) {
          dots.push({ x: p.x, y: Y(p), color: 'var(--viz-cat-1)', r: 4 });
          lines.push({ pts: [{ x: p.x, y: Y(p) }, { x: p.x, y: f.a * p.x ** -f.b }], color: 'var(--accent)', w: 1.5, op: 0.8 });
        }
        lines.push({ pts: [{ x: x0, y: f.a * x0 ** -f.b }, { x: x1, y: f.a * x1 ** -f.b }], color: 'var(--accent)', w: 2.5 });
        const searching = s.floor && t < 8;
        return {
          lines, dots, labels, phase: phaseAt(t, markers),
          readout: `${searching ? 'Trying floor' : s.floor ? 'Best floor' : 'No floor'} E = ${f.E.toFixed(3)}: exponent b = ${f.b.toFixed(4)}, typical miss ${f.rmse.toFixed(3)} nats, R² = ${f.r2.toFixed(4)}. Extended to 10¹⁵ FLOP it predicts ${predict(f, 1e15).toFixed(2)}.`,
          detail: searching ? 'Each frame is a real least-squares line for that candidate floor.' : 'Accent sticks show how far each dot sits from the line.',
        };
      }
      const f = full;
      const pred = (c: number) => predict(f, c);
      const fitEnd = used[used.length - 1].x;
      const reach = t < 3 ? fitEnd : t < 6 ? fitEnd * (x1 / fitEnd) ** ease((t - 3) / 3) : x1;
      const lo = used[0].x;
      lines.push({ pts: logspace(lo, fitEnd, 24).map((c) => ({ x: c, y: pred(c) })), color: 'var(--accent)', w: 2.5 });
      if (reach > fitEnd * 1.001) lines.push({ pts: logspace(fitEnd, reach, 40).map((c) => ({ x: c, y: pred(c) })), color: 'var(--accent)', w: 2.5, dash: true });
      for (const p of used) dots.push({ x: p.x, y: p.y, color: 'var(--viz-cat-1)', r: 4 });
      const done = t < 6 ? 0 : t >= 12 ? held.length : Math.min(held.length, Math.floor(((t - 6) / 6) * held.length) + 1);
      for (let i = 0; i < done; i++) {
        const p = held[i];
        lines.push({ pts: [{ x: p.x, y: p.y }, { x: p.x, y: pred(p.x) }], color: 'var(--viz-neg)', w: 1.5 });
        dots.push({ x: p.x, y: p.y, color: 'var(--viz-neg)', r: 4, hollow: true });
      }
      const lastShown = done > 0 ? held[done - 1] : null;
      let readout = `Fitted on the ${n} smallest budgets: ${s.floor && f.E > 0 ? `floor E = ${f.E.toFixed(3)}, ` : 'no floor, '}exponent b = ${f.b.toFixed(4)}.`;
      if (lastShown) {
        const e = pred(lastShown.x) - lastShown.y;
        readout += ` ${done === held.length ? 'Largest run' : `Held-out point ${done} of ${held.length}`}: predicted ${pred(lastShown.x).toFixed(3)}, measured ${lastShown.y.toFixed(3)}, error ${e >= 0 ? '+' : '−'}${Math.abs(e).toFixed(3)} nats.`;
      }
      return { lines, dots, labels, phase: phaseAt(t, markers), readout, detail: 'Filled dots were used for the fit; hollow dots were held out. Vertical sticks show prediction minus measurement.' };
    },
  };
}
/** Prediction error (predicted minus measured) for each held-out frontier point, for tests and prose. */
export function extrapolationErrors(data: Sweep, fitPoints: number, floor: boolean): { fit: Fit; errors: { x: number; predicted: number; measured: number; error: number }[] } {
  const env = frontier(data.runs), n = Math.max(3, Math.min(env.length - 3, fitPoints));
  const fit = fitPower(env.slice(0, n), floor);
  return { fit, errors: env.slice(n).map((p) => ({ x: p.x, predicted: predict(fit, p.x), measured: p.y, error: predict(fit, p.x) - p.y })) };
}

export function paramsDataScene(data: Sweep, s: Settings): Scene {
  const runs = data.runs, swap = s.swap;
  const keys = [...new Set(runs.map((r) => (swap ? r.N : r.D)))].sort((a, b) => a - b);
  const series = keys.map((key) => runs.filter((r) => (swap ? r.N : r.D) === key).sort((a, b) => (swap ? a.D - b.D : a.N - b.N)));
  const seg = 2, duration = seg * keys.length;
  const markers = keys.map((k, i) => ({ at: i * seg, label: swap ? compact(k) : compact(k) }));
  const x: Axis = swap
    ? { min: 6000, max: 3.5e5, log: true, label: 'Training characters D (log scale)', ticks: [1e4, 1e5].map((v) => ({ v, label: compact(v) })) }
    : { min: 2000, max: 1.7e5, log: true, label: 'Parameters N (log scale)', ticks: [3e3, 1e4, 3e4, 1e5].map((v) => ({ v, label: compact(v) })) };
  const y = lossAxis(2.0, 3.1, 0.2);
  return {
    mode: 'params-data', x, y, duration, markers, label: swap ? 'Loss against training characters, one curve per model size' : 'Loss against parameters, one curve per amount of training data',
    frame(t) {
      const lines: Line[] = [], dots: Dot[] = [], labels: Label[] = [];
      const cur = Math.min(keys.length - 1, Math.floor(t / seg));
      let curShown = 0;
      series.forEach((ser, i) => {
        const u = ease((t - i * seg) / (seg * 0.8));
        if (u <= 0) return;
        const pts = ser.map((r) => ({ x: swap ? r.D : r.N, y: r.validLoss }));
        const upto = u * (pts.length - 1), whole = Math.floor(upto + 1e-9), shown = pts.slice(0, whole + 1);
        if (i === cur) curShown = whole + 1;
        if (whole < pts.length - 1) { const f = upto - whole; shown.push({ x: pts[whole].x * (pts[whole + 1].x / pts[whole].x) ** f, y: pts[whole].y + (pts[whole + 1].y - pts[whole].y) * f }); }
        const col = cmix(keys.length > 1 ? i / (keys.length - 1) : 1);
        lines.push({ pts: shown, color: col, w: i === cur ? 2.8 : 1.6, op: i === cur ? 1 : 0.75 });
        pts.slice(0, whole + 1).forEach((p) => dots.push({ x: p.x, y: p.y, color: col, r: i === cur ? 3.6 : 2.6 }));
        if (u >= 0.999) labels.push({ x: pts[pts.length - 1].x, y: pts[pts.length - 1].y, text: compact(keys[i]), anchor: 'start', dx: 7, dy: 4, color: col });
      });
      const ser = series[cur], a = ser[0], b = ser[ser.length - 1], full = curShown >= ser.length;
      const subject = swap ? `${fmtInt(keys[cur])} parameters` : `${fmtInt(keys[cur])} training characters`;
      const at = (r: Run) => (swap ? `${fmtInt(r.D)} characters` : `${fmtInt(r.N)} parameters`);
      const readout = curShown === 0 ? 'Next curve starting.' : full
        ? `${subject}: loss ${a.validLoss.toFixed(4)} with ${at(a)}, ${b.validLoss.toFixed(4)} with ${at(b)}.`
        : `${subject}: loss ${a.validLoss.toFixed(4)} with ${at(a)}; the curve is still being drawn.`;
      return { lines, dots, labels, phase: phaseAt(t, markers), readout, detail: swap ? 'Curves run from the smallest to the largest model; labels give parameters.' : 'Curves run from the least to the most training data; labels give characters.' };
    },
  };
}

export const ISO_BUDGETS = [1e9, 3e9, 1e10, 3e10];
const ISO_COLORS = ['var(--viz-cat-1)', 'var(--viz-cat-2)', 'var(--viz-cat-3)', 'var(--viz-cat-4)'];
export function isoflopScene(data: Sweep, s: Settings): Scene {
  const sur = sharedSurface(data), user = 10 ** s.budget;
  const budgets = [...ISO_BUDGETS, user];
  const curveN = logspace(700, 1.5e6, 70);
  const x = sizeAxis(700, 1.5e6), y = lossAxis(2.0, 3.2, 0.2);
  const markers = [...ISO_BUDGETS.map((b, i) => ({ at: i * 2, label: sci(b, 1).replace('1.0×', '') })), { at: 8, label: 'Your budget' }];
  return {
    mode: 'isoflop', x, y, duration: 12, markers, label: 'Loss against model size at several fixed compute budgets',
    frame(t) {
      const lines: Line[] = [], dots: Dot[] = [], labels: Label[] = [];
      const shownCount = Math.min(budgets.length, Math.floor(t / 2) + 1), cur = Math.min(budgets.length - 1, Math.floor(t / 2));
      for (let i = 0; i < shownCount; i++) {
        const C = budgets[i], col = i === budgets.length - 1 ? 'var(--accent)' : ISO_COLORS[i], u = ease((t - i * 2) / 1.6);
        const pts = isoPoints(data.runs, C);
        const fitted = curveN.map((N) => ({ x: N, y: surfaceLoss(sur, N, C / (6 * N)) })).filter((p) => p.y >= y.min && p.y <= y.max);
        const opt = surfaceOptimum(sur, C);
        lines.push({ pts: fitted, color: col, w: i === cur ? 2 : 1.3, dash: true, op: 0.85 * u });
        if (pts.length > 1) lines.push({ pts: pts.map((p) => ({ x: p.N, y: p.loss })), color: col, w: i === cur ? 2.4 : 1.6, op: u });
        for (const p of pts) dots.push({ x: p.N, y: p.loss, color: col, r: i === cur ? 4 : 3, hollow: !p.measured });
        if (u > 0.95 && opt.loss >= y.min && opt.loss <= y.max && opt.N >= x.min && opt.N <= x.max) dots.push({ x: opt.N, y: opt.loss, color: col, r: 6, ring: true });
        if (u > 0.95 && pts.length) labels.push({ x: pts[pts.length - 1].N, y: pts[pts.length - 1].loss, text: i === budgets.length - 1 ? 'yours' : sci(C, 1), anchor: 'start', dx: 7, dy: -5, color: col });
      }
      const C = budgets[cur], pts = isoPoints(data.runs, C), opt = surfaceOptimum(sur, C);
      let bestMeasured = pts[0];
      for (const p of pts) if (p.loss < bestMeasured.loss) bestMeasured = p;
      const readout = `Budget ${sci(C, 2)} FLOP: best measured size ${fmtInt(bestMeasured.N)} parameters (loss ${bestMeasured.loss.toFixed(3)}, ${(bestMeasured.D / bestMeasured.N).toFixed(0)} characters per parameter). Fitted surface bottoms out near ${compact(opt.N)} parameters, ${(opt.D / opt.N).toFixed(0)} characters per parameter.`;
      return { lines, dots, labels, phase: phaseAt(t, markers), readout, detail: 'Hollow dots are read between a size’s two nearest data amounts; a solid dot would be a run at exactly this budget. Dashed: fitted surface. Ring: its minimum.' };
    },
  };
}

export const PUB_X = { min: 1e9, max: 1e13 }, PUB_Y = { min: 1e10, max: 1e14 };
export const PUB_LEVELS = [1.9, 2.0, 2.1, 2.2, 2.4, 2.7];
/** Points where the published fit equals `level`, as D against N. */
export function contour(level: number, p: Published, n = 60): Pt[] {
  const out: Pt[] = [];
  for (const N of logspace(PUB_X.min, PUB_X.max, n)) {
    const rest = level - p.E - p.A / N ** p.alpha;
    if (rest <= 0) continue;
    const D = (p.B / rest) ** (1 / p.beta);
    if (D >= PUB_Y.min && D <= PUB_Y.max) out.push({ x: N, y: D });
  }
  return out;
}
/** Budget for the published scene at time t: sweep up to the Chinchilla budget, hold there, then go on. */
export function publishedBudget(t: number): number {
  const lo = 1e21, hi = 3e24;
  if (t < 4) return lo * (CHINCHILLA_C / lo) ** ease(t / 4);
  if (t < 8) return CHINCHILLA_C;
  return CHINCHILLA_C * (hi / CHINCHILLA_C) ** ease((t - 8) / 4);
}
export function publishedScene(s: Settings): Scene {
  const p = PUBLISHED[s.pubFit];
  const x: Axis = { min: PUB_X.min, max: PUB_X.max, log: true, label: 'Parameters N (log scale)', ticks: [1e9, 1e10, 1e11, 1e12, 1e13].map((v) => ({ v, label: compact(v) })) };
  const y: Axis = { min: PUB_Y.min, max: PUB_Y.max, log: true, label: 'Training tokens D (log scale)', ticks: [1e10, 1e11, 1e12, 1e13, 1e14].map((v) => ({ v, label: compact(v) })) };
  const markers = [{ at: 0, label: 'Smaller budgets' }, { at: 4, label: 'Gopher’s budget' }, { at: 8, label: 'Larger budgets' }];
  const grid = logspace(1e21, 3e24, 50);
  const optLine = grid.map((c) => ({ x: optimalN(c, p), y: c / (6 * optimalN(c, p)) })).filter((q) => q.x >= x.min && q.x <= x.max && q.y >= y.min && q.y <= y.max);
  const kapLine = grid.map((c) => ({ x: kaplanN(c), y: c / (6 * kaplanN(c)) })).filter((q) => q.x >= x.min && q.x <= x.max && q.y >= y.min && q.y <= y.max);
  const levelLines: Line[] = PUB_LEVELS.map((lv) => ({ pts: contour(lv, p), color: 'var(--viz-seq-3)', w: 1, op: 0.9 }));
  const levelLabels: Label[] = PUB_LEVELS.flatMap((lv) => { const c = contour(lv, p); return c.length ? [{ x: c[c.length - 1].x, y: c[c.length - 1].y, text: lv.toFixed(1), anchor: 'end' as const, dx: -3, dy: -4, color: 'var(--text-muted)' }] : []; });
  return {
    mode: 'published', x, y, duration: 12, markers, label: 'Published loss map over parameters and tokens, with a compute budget line and two allocation rules',
    frame(t) {
      const C = publishedBudget(t), nO = optimalN(C, p), nK = kaplanN(C);
      const lines: Line[] = [...levelLines];
      lines.push({ pts: [{ x: x.min, y: C / (6 * x.min) }, { x: x.max, y: C / (6 * x.max) }], color: 'var(--text)', w: 1.8 });
      lines.push({ pts: optLine, color: 'var(--accent)', w: 2.2 });
      lines.push({ pts: kapLine, color: 'var(--viz-neg)', w: 2.2, dash: true });
      const dots: Dot[] = [];
      const inside = (N: number, D: number) => N >= x.min && N <= x.max && D >= y.min && D <= y.max;
      if (inside(nO, C / (6 * nO))) dots.push({ x: nO, y: C / (6 * nO), color: 'var(--accent)', r: 6 });
      if (inside(nK, C / (6 * nK))) dots.push({ x: nK, y: C / (6 * nK), color: 'var(--viz-neg)', r: 6 });
      for (const m of [CHINCHILLA, GOPHER]) dots.push({ x: m.N, y: m.D, color: 'var(--text)', r: 4.5, hollow: true });
      const labels: Label[] = [...levelLabels,
        { x: CHINCHILLA.N, y: CHINCHILLA.D, text: 'Chinchilla', anchor: 'start', dx: 8, dy: 4, color: 'var(--text)' },
        { x: GOPHER.N, y: GOPHER.D, text: 'Gopher', anchor: 'end', dx: -8, dy: -7, color: 'var(--text)' }];
      const lO = publishedLoss(nO, C / (6 * nO), p), lK = publishedLoss(nK, C / (6 * nK), p);
      const readout = `Budget ${sci(C, 3)} FLOP. Chinchilla-style: ${compact(nO)} parameters on ${compact(C / (6 * nO))} tokens, predicted loss ${lO.toFixed(3)}. Kaplan-style: ${compact(nK)} parameters on ${compact(C / (6 * nK))} tokens, predicted loss ${lK.toFixed(3)}.`;
      return { lines, dots, labels, phase: phaseAt(t, markers), readout, detail: `Published fit: ${p.short}. Lines are loss levels; the black line is the budget; hollow dots are the published Gopher and Chinchilla runs.` };
    },
  };
}

/** Tokens served as a function of time: none, then 10^9 growing to 10^13. */
export function servedAt(t: number): number {
  if (t < 3) return 0;
  return 10 ** (9 + 4 * ease((t - 3) / 7));
}
export function overtrainTarget(p: Published) { return publishedLoss(CHINCHILLA.N, CHINCHILLA.D, p); }
export function overtrainScene(s: Settings): Scene {
  const p = PUBLISHED[s.pubFit], target = overtrainTarget(p), lo = minParams(target, p) * 1.02;
  const x: Axis = { min: 5e9, max: 5e11, log: true, label: 'Parameters N (log scale)', ticks: [1e10, 1e11].map((v) => ({ v, label: compact(v) })) };
  const y: Axis = { min: 4e23, max: 1.5e25, log: true, label: 'Total FLOP: training plus serving (log scale)', ticks: [5e23, 1e24, 2e24, 5e24, 1e25].map((v) => ({ v, label: sci(v, 1).replace('1.0×', '') })) };
  const markers = [{ at: 0, label: 'No serving' }, { at: 3, label: 'Serving grows' }, { at: 10, label: 'Heavy use' }];
  const Ns = logspace(Math.max(lo, x.min), x.max, 70);
  const base: Line = { pts: Ns.map((N) => ({ x: N, y: totalCost(N, target, 0, p) })).filter((q) => Number.isFinite(q.y)), color: 'var(--text-muted)', w: 1.5, dash: true };
  const zero = servingOptimum(0, target, p);
  return {
    mode: 'overtrain', x, y, duration: 12, markers, label: 'Total compute against model size for a fixed target loss, growing with the number of tokens served',
    frame(t) {
      const served = servedAt(t), opt = servingOptimum(served, target, p);
      const pts = Ns.map((N) => ({ x: N, y: totalCost(N, target, served, p) })).filter((q) => Number.isFinite(q.y) && q.y >= y.min && q.y <= y.max);
      const lines: Line[] = [base, { pts, color: 'var(--accent)', w: 2.5 }];
      const dots: Dot[] = [{ x: zero.N, y: zero.cost, color: 'var(--text-muted)', r: 4.5, hollow: true }, { x: opt.N, y: opt.cost, color: 'var(--accent)', r: 6 }];
      const labels: Label[] = [{ x: zero.N, y: zero.cost, text: 'training only', anchor: 'start', dx: 8, dy: 14, color: 'var(--text-muted)' }];
      const readout = `${served === 0 ? 'No tokens served' : `${compact(served)} tokens served`}: cheapest model has ${compact(opt.N, 3)} parameters, trained on ${compact(opt.D, 3)} tokens, ${opt.ratio.toFixed(0)} tokens per parameter. Target loss ${target.toFixed(3)}.`;
      return { lines, dots, labels, phase: phaseAt(t, markers), readout, detail: `Published fit: ${p.short}. Total FLOP = 6ND + 2N × tokens served. The target is the loss the fit predicts for Chinchilla 70B on 1.4T tokens.` };
    },
  };
}

export function makeScene(mode: Mode, data: Sweep | null, s: Settings): Scene | null {
  if (mode === 'published') return publishedScene(s);
  if (mode === 'overtrain') return overtrainScene(s);
  if (!data) return null;
  if (mode === 'sweep') return sweepScene(data);
  if (mode === 'compute' || mode === 'extrapolate') return fitScene(mode, data, s);
  if (mode === 'params-data') return paramsDataScene(data, s);
  return isoflopScene(data, s);
}
export const frameAt = (t: number, scene: Scene, hover = -1): Frame => scene.frame(Math.min(scene.duration, Math.max(0, t)), hover);
export const TITLES: Record<Mode, string> = {
  sweep: 'Forty-two training runs', compute: 'A power law through the best runs', 'params-data': 'Loss depends on model size and on data',
  isoflop: 'Same compute, different model sizes', published: 'Published fit: where to spend a compute budget', extrapolate: 'Fit on small runs, predict larger ones',
  overtrain: 'Training cost plus serving cost',
};
