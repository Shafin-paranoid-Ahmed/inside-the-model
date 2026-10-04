/** Trains one model (with or without attention) off the main thread. One message in, progress and one result out. */
import { train, type Run } from './sim';

export type FromWorker = { kind: 'progress'; frac: number } | { kind: 'done'; run: Run };

self.onmessage = (e: MessageEvent<{ attn: boolean }>) => {
  const post = (m: FromWorker) => (self as unknown as Worker).postMessage(m);
  post({ kind: 'done', run: train(e.data.attn, undefined, undefined, (frac) => post({ kind: 'progress', frac })) });
};
