/// <reference lib="webworker" />
// Long generations re-read the last 64 characters at every step, so they run here, not on the page's thread.
import { loadModel } from '../../../lib/transformer';
import { encode } from '../../../lib/lm/charset';
import { LAB_N, TEMPS, labOptions, labPrompt, makeGreedy, stepper, tempRun, uniforms, type GreedyData, type LabSettings, type Run } from './sim';
import type { SampleStep } from '../../../lib/lm/sample';

export type ToWorker = { id: number; kind: 'greedy'; seed: number } | { id: number; kind: 'temp'; seed: number } | { id: number; kind: 'lab'; settings: LabSettings };
export type FromWorker =
  | { id: number; kind: 'greedy'; data: GreedyData }
  | { id: number; kind: 'temp'; i: number; text: string; mean: number }
  | { id: number; kind: 'lab'; from: number; steps: SampleStep[] }
  | { id: number; kind: 'error' };

let latest = 0, greedyLane: Run | undefined;
const post = (m: FromWorker) => (self as unknown as Worker).postMessage(m);
const yieldToQueue = () => new Promise<void>((r) => setTimeout(r, 0)); // lets a newer request replace this one

self.onmessage = async (e: MessageEvent<ToWorker>) => {
  const m = e.data;
  latest = m.id;
  try {
    const w = await loadModel('house');
    if (m.kind === 'greedy') {
      const data = makeGreedy(w, m.seed, undefined, greedyLane);
      greedyLane = data.greedy;
      if (latest === m.id) post({ id: m.id, kind: 'greedy', data });
    } else if (m.kind === 'temp') {
      for (let i = 0; i < TEMPS.length; i++) {
        const r = tempRun(w, TEMPS[i], m.seed);
        if (latest !== m.id) return;
        post({ id: m.id, kind: 'temp', i, ...r });
        await yieldToQueue();
        if (latest !== m.id) return;
      }
    } else {
      const st = stepper(w, encode(labPrompt(m.settings)), labOptions(m.settings, w.config.vocab.length)), us = uniforms(m.settings.seed, LAB_N);
      for (let from = 0; from < LAB_N; from += 12) {
        const steps = us.slice(from, from + 12).map((u) => st.next(u));
        if (latest !== m.id) return;
        post({ id: m.id, kind: 'lab', from, steps });
        await yieldToQueue();
        if (latest !== m.id) return;
      }
    }
  } catch {
    post({ id: m.id, kind: 'error' });
  }
};
