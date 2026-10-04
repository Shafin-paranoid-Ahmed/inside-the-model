/** Trains everything off the main thread. One message in, one result out. */
import { computeCurves, computeInspect, type Curves, type Inspect, type DatasetId } from './sim';

export type ToWorker = { kind: DatasetId | 'inspect' };
export type FromWorker = { kind: 'tabular' | 'pixels'; curves: Curves } | { kind: 'inspect'; inspect: Inspect };

self.onmessage = (e: MessageEvent<ToWorker>) => {
  const k = e.data.kind;
  const out: FromWorker = k === 'inspect' ? { kind: k, inspect: computeInspect() } : { kind: k, curves: computeCurves(k) };
  (self as unknown as Worker).postMessage(out);
};
