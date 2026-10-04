/** Trains the network and runs every analysis off the main thread. One message in, progress and one result out. */
import { trainAll, type Trained } from './sim';

export type FromWorker = { kind: 'progress'; frac: number } | { kind: 'done'; trained: Trained };

self.onmessage = () => {
  const post = (m: FromWorker) => (self as unknown as Worker).postMessage(m);
  post({ kind: 'done', trained: trainAll(undefined, undefined, (frac) => post({ kind: 'progress', frac })) });
};
