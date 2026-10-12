/** Trains the noise-prediction network off the main thread, then samples from it. One message in, progress and one result out. */
import { trainEps, reversePath, netScoreFn, startNoise, REVERSE_STEPS, MOONS, type TrainRun } from './sim';

export type FromWorker = { kind: 'progress'; frac: number } | { kind: 'done'; run: TrainRun };

self.onmessage = () => {
  const post = (m: FromWorker) => (self as unknown as Worker).postMessage(m);
  const run = trainEps(undefined, (f) => post({ kind: 'progress', frac: 0.7 * f }));
  const sf = netScoreFn(run.snaps[run.snaps.length - 1]), xT = startNoise();
  run.paths = { ode: reversePath(MOONS, xT, 'ode', REVERSE_STEPS, 5, sf) };
  post({ kind: 'progress', frac: 0.85 });
  run.paths.sde = reversePath(MOONS, xT, 'sde', REVERSE_STEPS, 5, sf);
  post({ kind: 'done', run });
};
