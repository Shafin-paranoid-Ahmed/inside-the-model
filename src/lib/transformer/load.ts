import { parameterLayout, type Config, type Offsets, type Weights } from './model';

export type DType = 'float16' | 'int8';
export interface Manifest {
  version: 1;
  name: string;
  config: Config;
  vocab: readonly string[];
  offsets: Offsets;
  dtype: DType;
  parameterCount: number;
  sha256?: string;
  scales?: Record<string, number[]>;
  metrics: Record<string, unknown>;
  lossCurve: { step: number; loss: number }[];
  training: Record<string, unknown>;
}
const roundEven = (n: number) => {
  const lo = Math.floor(n), rest = n - lo;
  return rest > 0.5 || (rest === 0.5 && lo % 2 === 1) ? lo + 1 : lo;
};
/** IEEE binary16 round-to-nearest, ties-to-even, directly from float64 (no float32 double rounding). */
export function toFloat16(value: number): number {
  const sign = value < 0 || Object.is(value, -0) ? 0x8000 : 0, x = Math.abs(value);
  if (Number.isNaN(x)) return 0x7e00;
  if (x >= 65520) return sign | 0x7c00;
  if (x < 2 ** -14) return sign | roundEven(x / (2 ** -24));
  let exponent = Math.floor(Math.log2(x));
  let fraction = roundEven((x / (2 ** exponent) - 1) * 1024);
  if (fraction === 1024) { fraction = 0; exponent++; }
  return sign | ((exponent + 15) << 10) | fraction;
}
export function fromFloat16(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1, exponent = bits >> 10 & 31, fraction = bits & 1023;
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * (exponent ? (1 + fraction / 1024) * (2 ** (exponent - 15)) : fraction * (2 ** -24));
}
export function encodeWeights(w: Weights, dtype: DType = w.params.length * 2 > 40960 ? 'int8' : 'float16'): {
  bytes: Uint8Array; dtype: DType; scales?: Record<string, number[]>;
} {
  if (w.params.some((x) => !Number.isFinite(x))) throw new Error('Cannot encode non-finite weights');
  const bytes = new Uint8Array(w.params.length * (dtype === 'float16' ? 2 : 1));
  if (dtype === 'float16') {
    const view = new DataView(bytes.buffer);
    w.params.forEach((x, i) => {
      const bits = toFloat16(x);
      if ((bits & 0x7c00) === 0x7c00) throw new Error('Weight overflows float16');
      view.setUint16(i * 2, bits, true);
    });
    return { bytes, dtype };
  }
  const scales: Record<string, number[]> = {};
  const signed = new Int8Array(bytes.buffer);
  for (const [name, tensor] of Object.entries(w.offsets)) {
    const rowScales: number[] = [];
    for (let r = 0; r < tensor.rows; r++) {
      const start = tensor.offset + r * tensor.cols;
      let max = 0;
      for (let j = 0; j < tensor.cols; j++) max = Math.max(max, Math.abs(w.params[start + j]));
      const scale = max ? max / 127 : 1; rowScales.push(scale);
      for (let j = 0; j < tensor.cols; j++) signed[start + j] = Math.max(-127, Math.min(127, Math.round(w.params[start + j] / scale)));
    }
    scales[name] = rowScales;
  }
  return { bytes, dtype, scales };
}

export function parseModel(manifest: Manifest, bytes: Uint8Array): Weights {
  const layout = parameterLayout(manifest.config);
  if (manifest.version !== 1 || manifest.parameterCount !== layout.size ||
    JSON.stringify(manifest.offsets) !== JSON.stringify(layout.offsets) ||
    JSON.stringify(manifest.vocab) !== JSON.stringify(manifest.config.vocab)) throw new Error('Invalid model manifest/layout');
  if (manifest.dtype !== 'float16' && manifest.dtype !== 'int8') throw new Error('Unsupported dtype');
  if (bytes.byteLength !== layout.size * (manifest.dtype === 'float16' ? 2 : 1)) throw new Error('Model byte length differs from manifest');
  const params = new Float64Array(layout.size);
  if (manifest.dtype === 'float16') {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    for (let i = 0; i < params.length; i++) params[i] = fromFloat16(view.getUint16(i * 2, true));
  } else {
    const signed = new Int8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    for (const [name, tensor] of Object.entries(layout.offsets)) {
      const scales = manifest.scales?.[name];
      if (!scales || scales.length !== tensor.rows || scales.some((s) => !Number.isFinite(s) || s <= 0)) throw new Error(`Invalid row scales: ${name}`);
      for (let r = 0; r < tensor.rows; r++) for (let j = 0; j < tensor.cols; j++) {
        const i = tensor.offset + r * tensor.cols + j; params[i] = signed[i] * scales[r];
      }
    }
  }
  if (params.some((p) => !Number.isFinite(p))) throw new Error('Non-finite decoded weights');
  return { config: manifest.config, params, offsets: layout.offsets };
}

const loaded = new Map<string, Promise<Weights>>();
/** Call when an island becomes visible. No fetch at import time; share one promise per model. */
export function loadModel(name: string): Promise<Weights> {
  if (!/^[a-z0-9][a-z0-9-]*$/.test(name)) return Promise.reject(new Error('Invalid model name'));
  const existing = loaded.get(name);
  if (existing) return existing;
  const pending = (async () => {
    const [json, binary] = await Promise.all([fetch(`/models/${name}.json`), fetch(`/models/${name}.bin`)]);
    if (!json.ok || !binary.ok) throw new Error(`Could not load model ${name}`);
    const manifest: Manifest = await json.json(), bytes = new Uint8Array(await binary.arrayBuffer());
    if (manifest.name !== name) throw new Error('Model name differs from requested model');
    if (manifest.sha256) {
      const digest = await crypto.subtle.digest('SHA-256', bytes);
      const hex = Array.from(new Uint8Array(digest), (v) => v.toString(16).padStart(2, '0')).join('');
      if (hex !== manifest.sha256) throw new Error('Model checksum mismatch');
    }
    return parseModel(manifest, bytes);
  })();
  loaded.set(name, pending);
  // A transient request failure may be retried when another visible island requests it.
  pending.catch(() => { if (loaded.get(name) === pending) loaded.delete(name); });
  return pending;
}
