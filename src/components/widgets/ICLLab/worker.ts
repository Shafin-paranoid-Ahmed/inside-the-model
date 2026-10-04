import { loadModel } from '../../../lib/transformer/load';
import { COUNTS, FORMATS, ablations, accuracy } from './sim';

// One worker per page. Each request is answered separately, so a figure can draw as soon as its own numbers are ready.
export type Request = { id: number; kind: 'rule' | 'random' | 'formats' | 'ablations' };
const weights = loadModel('icl');

self.onmessage = async ({ data }: MessageEvent<Request>) => {
  try {
    const w = await weights;
    const result = data.kind === 'rule' ? COUNTS.map((k) => accuracy(w, k))
      : data.kind === 'random' ? COUNTS.map((k) => accuracy(w, k, { random: true }))
      : data.kind === 'formats' ? FORMATS.map((f) => accuracy(w, 4, { format: f.id }))
      : ablations(w);
    (self as unknown as Worker).postMessage({ id: data.id, data: result });
  } catch (error) { (self as unknown as Worker).postMessage({ id: data.id, error: String(error) }); }
};
