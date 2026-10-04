/// <reference lib="webworker" />
import { Trainer, type Snapshot } from './trainer';

export type ToWorker =
  | { type: 'reset'; seed: number; xs: Float32Array; ys: Float32Array }
  | { type: 'data'; xs: Float32Array; ys: Float32Array }
  | { type: 'run'; on: boolean };
export type { Snapshot };

const STEPS_PER_TICK = 2, TICK_MS = 40; // ~25 snapshots/s
let trainer: Trainer | undefined;
let timer = 0;

function tick() {
  if (!trainer) return;
  for (let i = 0; i < STEPS_PER_TICK; i++) trainer.train();
  const s = trainer.snapshot();
  (self as unknown as Worker).postMessage(s, [s.weights.buffer, s.grads.buffer]);
}

self.onmessage = (e: MessageEvent<ToWorker>) => {
  const m = e.data;
  if (m.type === 'reset') {
    trainer = new Trainer(m.seed, { xs: m.xs, ys: m.ys });
    tick();
  } else if (m.type === 'data') {
    if (trainer) trainer.data = { xs: m.xs, ys: m.ys };
  } else {
    clearInterval(timer);
    if (m.on) timer = self.setInterval(tick, TICK_MS);
  }
};
