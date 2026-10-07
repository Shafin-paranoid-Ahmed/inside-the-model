/// <reference lib="webworker" />
// Heavy Day 11 and Day 12 runs (DQN comparisons, many-run curves, PPO/RLHF training, DPO, GRPO, threshold scans) run
// here, off the page's thread. Any exported function of src/lib/rl or src/lib/align can be called by name (the two
// share no export names); arguments and results are structured clones.
import * as rl from '../../../lib/rl';
import * as align from '../../../lib/align';

const fns = { ...align, ...rl } as unknown as Record<string, unknown>;
self.onmessage = (e: MessageEvent<{ id: number; fn: string; args: unknown[] }>) => {
  const { id, fn, args } = e.data, f = fns[fn];
  try {
    if (typeof f !== 'function') throw new Error(`src/lib/rl and src/lib/align have no function ${fn}`);
    (self as unknown as Worker).postMessage({ id, result: f(...args) });
  } catch (err) {
    (self as unknown as Worker).postMessage({ id, error: String(err) });
  }
};
