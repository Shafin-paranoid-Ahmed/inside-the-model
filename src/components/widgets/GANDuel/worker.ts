/** Trains one GAN run off the main thread. One message in, progress and one result out. */
import { train, type GanRun, type RunName } from './sim';

export type FromWorker = { kind: 'progress'; frac: number } | { kind: 'done'; run: GanRun };

self.onmessage = (e: MessageEvent<{ name: RunName }>) => {
  const post = (m: FromWorker) => (self as unknown as Worker).postMessage(m);
  post({ kind: 'done', run: train(e.data.name, (frac) => post({ kind: 'progress', frac })) });
};
