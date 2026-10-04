import { loadModel } from '../../../lib/transformer/load';
import { ask, runExperiment, type Person } from './sim';

export type Request = { id: number; kind: 'experiment' } | { id: number; kind: 'ask'; name: string; stated?: { name: string; month: number } } | { id: number; kind: 'stated'; name: string };
const weights = loadModel('facts');
const people = fetch('/models/facts-data.json').then((r) => r.json()).then((d): Person[] => d.people.map((p: Person) => ({ name: p.name, month: p.month, count: p.count })));

self.onmessage = async ({ data }: MessageEvent<Request>) => {
  try {
    const w = await weights;
    if (data.kind === 'experiment') { const p = await people; (self as unknown as Worker).postMessage({ id: data.id, data: { people: p, exp: runExperiment(w, p) } }); }
    else if (data.kind === 'ask') (self as unknown as Worker).postMessage({ id: data.id, data: ask(w, data.name, data.stated) });
    else (self as unknown as Worker).postMessage({ id: data.id, data: Array.from({ length: 12 }, (_, m) => ask(w, data.name, { name: data.name, month: m })) });
  } catch (error) { (self as unknown as Worker).postMessage({ id: data.id, error: String(error) }); }
};
