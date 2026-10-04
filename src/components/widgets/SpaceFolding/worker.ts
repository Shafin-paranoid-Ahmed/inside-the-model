import { compute, type Run, type RunId } from './sim';
export type Reply = { id: RunId; run: Run } | { id: RunId; error: string };
self.onmessage = (e: MessageEvent<{ id: RunId }>) => {
  try { self.postMessage({ id: e.data.id, run: compute(e.data.id) } satisfies Reply); }
  catch (error) { self.postMessage({ id: e.data.id, error: String(error) } satisfies Reply); }
};
