/** Trains one model off the main thread. One message in (which model), progress messages, one result out. */
import { train, type ModelId, type Run } from './sim';

export type FromWorker = { kind: 'progress'; frac: number } | { kind: 'done'; run: Run };

self.onmessage = (e: MessageEvent<{ id: ModelId }>) => {
  const post = (m: FromWorker) => (self as unknown as Worker).postMessage(m);
  post({ kind: 'done', run: train(e.data.id, undefined, (frac) => post({ kind: 'progress', frac })) });
};
