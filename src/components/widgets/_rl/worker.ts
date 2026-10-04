/// <reference lib="webworker" />
// Heavy Day 11 runs (DQN comparisons, many-run curves, PPO training, policy threshold scans) run here, off the page's
// thread. Any exported function of src/lib/rl can be called by name; arguments and results are structured clones.
import * as rl from '../../../lib/rl';

const fns = rl as unknown as Record<string, unknown>;
self.onmessage = (e: MessageEvent<{ id: number; fn: string; args: unknown[] }>) => {
  const { id, fn, args } = e.data, f = fns[fn];
  try {
    if (typeof f !== 'function') throw new Error(`src/lib/rl has no function ${fn}`);
    (self as unknown as Worker).postMessage({ id, result: f(...args) });
  } catch (err) {
    (self as unknown as Worker).postMessage({ id, error: String(err) });
  }
};
