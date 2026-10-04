/** Trains one network off the main thread. One message in, one result out. */
import { runNet, type NetId, type NetRun } from './sim';

export type ToWorker = { id: NetId; seed: number };
export type FromWorker = { id: NetId; seed: number; run: NetRun };

self.onmessage = (e: MessageEvent<ToWorker>) => {
  const { id, seed } = e.data;
  const run = runNet(id, seed);
  (self as unknown as Worker).postMessage({ id, seed, run } satisfies FromWorker, [run.loss.buffer, run.snaps.buffer]);
};
