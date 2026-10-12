/** Trains the velocity network off the main thread, then samples from it for each step count. One message in, progress and one result out. */
import { trainFlow, samplePaths, type FlowRun } from './sim';

export type FromWorker = { kind: 'progress'; frac: number } | { kind: 'done'; run: FlowRun };

self.onmessage = () => {
  const post = (m: FromWorker) => (self as unknown as Worker).postMessage(m);
  const run = trainFlow(undefined, (f) => post({ kind: 'progress', frac: 0.85 * f }));
  post({ kind: 'progress', frac: 0.9 });
  run.paths = samplePaths(run.snaps[run.snaps.length - 1]);
  post({ kind: 'done', run });
};
