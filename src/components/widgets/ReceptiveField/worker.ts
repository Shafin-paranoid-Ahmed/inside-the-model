import { buildEffective } from './sim';
export type Reply = ReturnType<typeof buildEffective> | { error: string };
self.onmessage = () => {
  try { self.postMessage(buildEffective() satisfies Reply); }
  catch (error) { self.postMessage({ error: String(error) } satisfies Reply); }
};
