import { computeRun } from './sim';
self.onmessage = () => {
  try { self.postMessage({ run: computeRun() }); }
  catch (error) { self.postMessage({ error: String(error) }); }
};
