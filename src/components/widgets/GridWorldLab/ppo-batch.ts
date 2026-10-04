/** Shared by the PPO grid modes: a batch collected from a policy that PPO has already trained for a few iterations. */
import { createRng } from '../../../lib/math/rng';
import { makeBatch, trainPPO } from '../../../lib/rl/ppo';
import { int, key, num, type Kit, type Opts } from '../_rl/kit';
import { gridOf } from './sim';

/** `warm` PPO iterations (seed `seed`, lambda 0.95) give a policy whose batch has advantages of both signs. */
export function warmBatch(o: Opts, kit: Kit, warm: number, seed0: number) {
  const g = gridOf(o, 0.9), seed = int(o.seed, seed0), lambda = num(o.lambda, 0.95), n = int(o.batch, 64);
  return kit.memo(key('ppo-warm', g.stepReward, g.slip, g.gamma, seed, lambda, n, warm), () => {
    const w = trainPPO(g, { iterations: warm, seed, gamma: g.gamma, lambda: 0.95 });
    return { g, seed, lambda, b: makeBatch(g, w.theta, w.V, n, createRng(seed), g.gamma, lambda), theta: w.theta, V: w.V };
  });
}
