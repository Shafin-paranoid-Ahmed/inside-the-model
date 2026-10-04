/**
 * Pure logic for ReceptiveField.
 *
 * Part 1 (modes one / stack / stride): a chain of 1-D layers, a cross-section through the image. Layer l has
 * n_l units; unit i of layer l looks at k units of layer l-1 starting at i * stride (no padding). The
 * receptive field of a unit is traced backward layer by layer with the recurrences
 *   r_l = r_{l-1} + (k_l - 1) * j_{l-1}       (field size on the input)
 *   j_l = j_{l-1} * s_l                        (spacing of layer-l units, in input pixels)
 *
 * Part 2 (mode effective): a real seeded stack of 3x3 conv + ReLU layers built with the shared CNN core. The
 * gradient of one centre output unit is backpropagated to the input with conv2dBackward / reluBackward.
 */
import { conv2d, conv2dBackward, convLayer, map3, outSize, relu, reluBackward, type ConvLayer, type Map3 } from '../../../lib/cnn/conv';
import { createRng } from '../../../lib/math/rng';

export type Mode = 'one' | 'stack' | 'stride' | 'effective';

/* ---------------------------------------------------------------- part 1 */

export interface LayerSpec { k: number; stride: number; label: string }
export interface Scene { specs: LayerSpec[]; n: number }
export interface Pick { col: number; row: number }

const conv3 = (): LayerSpec => ({ k: 3, stride: 1, label: 'conv 3×3' });
const pool2 = (): LayerSpec => ({ k: 2, stride: 2, label: 'pool 2×2' });

/** `plain` replaces the pooling layer of the stride scene by an ordinary conv 3x3 (same number of layers). */
export function buildScene(mode: Exclude<Mode, 'effective'>, plain = false): Scene {
  if (mode === 'one') return { specs: [conv3()], n: 7 };
  if (mode === 'stack') return { specs: Array.from({ length: 6 }, conv3), n: 17 };
  return { specs: [conv3(), plain ? conv3() : pool2(), conv3(), conv3(), conv3()], n: 21 };
}

/** Unit counts per layer, index 0 = input. */
export function sizes(sc: Scene): number[] {
  const out = [sc.n];
  for (const s of sc.specs) out.push(outSize(out[out.length - 1], s.k, s.stride));
  return out;
}

/** Field size r and spacing j of a layer-l unit, in input pixels; index 0 = input (r = 1, j = 1). */
export function rfTable(specs: LayerSpec[]): { r: number; jump: number }[] {
  const t = [{ r: 1, jump: 1 }];
  for (const s of specs) {
    const p = t[t.length - 1];
    t.push({ r: p.r + (s.k - 1) * p.jump, jump: p.jump * s.stride });
  }
  return t;
}

/** Closed form for a stack of identical stride-1 layers: 1 + L (k - 1). */
export const stackRf = (layers: number, k = 3) => 1 + layers * (k - 1);

/** Index range [lo, hi] of the units each layer contributes to the field of top unit `u`. */
export function region(sc: Scene, u: number): [number, number][] {
  const L = sc.specs.length, out: [number, number][] = new Array(L + 1);
  out[L] = [u, u];
  for (let l = L; l >= 1; l--) {
    const { k, stride } = sc.specs[l - 1], [lo, hi] = out[l];
    out[l - 1] = [lo * stride, hi * stride + k - 1];
  }
  return out;
}

/** Top unit that sits closest to input pixel `px` (used for clicks on the image panel). */
export function unitNear(sc: Scene, px: number): number {
  const top = rfTable(sc.specs)[sc.specs.length], n = sizes(sc)[sc.specs.length];
  const i = Math.round((px - (top.r - 1) / 2) / top.jump);
  return Math.min(n - 1, Math.max(0, i));
}

export const centerPick = (sc: Scene): Pick => {
  const n = sizes(sc)[sc.specs.length];
  return { col: n >> 1, row: n >> 1 };
};

const PICK = 1.2, PER = 1, HOLD = 1.8;
export const durationOf = (sc: Scene) => PICK + PER * sc.specs.length + HOLD;
export const markersOf = (sc: Scene) => [
  { at: 0, label: 'Pick' },
  { at: PICK, label: 'Trace back' },
  { at: PICK + PER * sc.specs.length, label: 'Field' },
];
const ease = (p: number) => { const q = Math.min(1, Math.max(0, p)); return q * q * (3 - 2 * q); };

export interface Frame {
  phase: 'pick' | 'trace' | 'done';
  /** Layers crossed so far (rows shown by the drawing). */
  back: number;
  /** reveal[l] 0..1 of the traced region in layer l (the top layer is always 1). */
  reveal: number[];
  /** Pulse 0..1 of the chosen unit while picking. */
  pulse: number;
}

export function frameAt(t: number, sc: Scene): Frame {
  const L = sc.specs.length;
  const front = Math.min(L, Math.max(0, (t - PICK) / PER));
  const reveal = Array.from({ length: L + 1 }, (_, l) => (l === L ? 1 : ease(front - (L - l - 1))));
  return {
    phase: t < PICK ? 'pick' : t < PICK + PER * L ? 'trace' : 'done',
    // layers crossed = lower rows already shown (the drawing and the readout both use reveal > 0.5)
    back: reveal.slice(0, L).filter((v) => v > 0.5).length,
    reveal,
    pulse: t < PICK ? 0.5 - 0.5 * Math.cos((t / PICK) * Math.PI * 2) : 0,
  };
}

/** What the readout says at this frame: layer and cells seen in the deepest fully-traced layer, and the final field. */
export function readoutAt(sc: Scene, pick: Pick, fr: Frame) {
  const L = sc.specs.length, reg = region(sc, pick.col), layer = L - fr.back;
  return { layer, cells: reg[layer][1] - reg[layer][0] + 1, field: rfTable(sc.specs)[L].r };
}

/* ---------------------------------------------------------------- part 2 */

export const EFF = { n: 33, maxDepth: 12, channels: 4, images: 6, seed: 5 };
export interface EffNet { layers: ConvLayer[]; images: Map3[] }

/** Seeded random stack (He-scaled 3x3 weights, zero bias) and a few random input images. */
export function makeEffNet(cfg = EFF): EffNet {
  const rng = createRng(cfg.seed);
  const layers: ConvLayer[] = [];
  for (let d = 0; d < cfg.maxDepth; d++) {
    const L = convLayer(d === 0 ? 1 : cfg.channels, cfg.channels, 3, 1, 1);
    const std = Math.sqrt(2 / (9 * L.inC));
    for (let i = 0; i < L.weight.length; i++) L.weight[i] = rng.normal(0, std);
    layers.push(L);
  }
  const images = Array.from({ length: cfg.images }, () => {
    const m = map3(1, cfg.n, cfg.n);
    for (let i = 0; i < m.data.length; i++) m.data[i] = rng.normal();
    return m;
  });
  return { layers, images };
}

/** Forward pass of every layer for one image: xs[d] is the input of layer d (post-ReLU), pre[d] its pre-activation. */
export function forwardAll(net: EffNet, img: Map3) {
  const xs: Map3[] = [img], pre: Map3[] = [];
  for (let d = 0; d < net.layers.length; d++) {
    const z = conv2d(xs[d], net.layers[d]);
    pre.push(z);
    xs.push(relu(z));
  }
  return { xs, pre };
}

/**
 * Mean |d output(centre, channel 0) / d input| over the images, for a network made of the first `depth` layers.
 * The output is the last conv's pre-activation (no ReLU after it), so layers 1..depth-1 are shared with deeper
 * networks and `fwd` (one forwardAll per image) can be reused across depths.
 */
export function effectiveMap(net: EffNet, depth: number, n = EFF.n, fwd = net.images.map((im) => forwardAll(net, im))): Float64Array {
  const acc = new Float64Array(n * n);
  const c = n >> 1;
  fwd.forEach(({ xs, pre }) => {
    let g = map3(net.layers[depth - 1].outC, n, n);
    g.data[c * n + c] = 1; // channel 0, centre pixel
    for (let d = depth - 1; d >= 0; d--) {
      const L = net.layers[d];
      g = conv2dBackward(xs[d], L, g, new Float64Array(L.weight.length), new Float64Array(L.outC));
      if (d > 0) g = reluBackward(pre[d - 1], g);
    }
    for (let i = 0; i < acc.length; i++) acc[i] += Math.abs(g.data[i]) / fwd.length;
  });
  return acc;
}

export interface EffStats {
  depth: number;
  /** Side of the theoretical receptive field, 2 * depth + 1. */
  theory: number;
  /** Side of the smallest centred square holding 90 percent of the gradient mass. */
  core: number;
}

/** Cumulative gradient mass by square ring around the centre (index = distance from the centre). */
export function ringMass(map: Float64Array, n = EFF.n): Float64Array {
  const c = n >> 1, rings = new Float64Array(c + 1);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) rings[Math.max(Math.abs(x - c), Math.abs(y - c))] += map[y * n + x];
  for (let r = 1; r <= c; r++) rings[r] += rings[r - 1];
  return rings;
}

export function effStats(map: Float64Array, depth: number, frac = 0.9, n = EFF.n): EffStats {
  const rings = ringMass(map, n), total = rings[rings.length - 1];
  let r = 0;
  while (rings[r] < frac * total) r++;
  return { depth, theory: 2 * depth + 1, core: 2 * r + 1 };
}

/** Column marginal: sum over rows of |gradient| for every input column. */
export function marginal(map: Float64Array, n = EFF.n): Float64Array {
  const m = new Float64Array(n);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) m[x] += map[y * n + x];
  return m;
}

/** Everything precomputed once: maps, stats and column profiles for depths 1..maxDepth (index depth - 1). */
export function buildEffective(cfg = EFF) {
  const net = makeEffNet(cfg), fwd = net.images.map((im) => forwardAll(net, im));
  const maps: Float64Array[] = [], stats: EffStats[] = [], margs: Float64Array[] = [];
  for (let d = 1; d <= cfg.maxDepth; d++) {
    const m = effectiveMap(net, d, cfg.n, fwd);
    maps.push(m); stats.push(effStats(m, d, 0.9, cfg.n)); margs.push(marginal(m, cfg.n));
  }
  return { maps, stats, margs };
}

const EFF_STEP = 0.8, EFF_HOLD = 1.6;
export const EFF_T = EFF.maxDepth * EFF_STEP + EFF_HOLD;
export const EFF_MARKERS = [
  { at: 0, label: 'Layers 1–3' },
  { at: 3 * EFF_STEP, label: '4–7' },
  { at: 7 * EFF_STEP, label: '8–11' },
  { at: 11 * EFF_STEP, label: '12' },
];
/** Depth (1..maxDepth) shown at time t. Everything shown at that time belongs to one real network. */
export const effDepthAt = (t: number) => Math.min(EFF.maxDepth, 1 + Math.floor(Math.max(0, t) / EFF_STEP));
