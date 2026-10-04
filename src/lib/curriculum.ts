import { getCollection, type CollectionEntry } from 'astro:content';

type Lesson = CollectionEntry<'lessons'>;

export const TRACKS = [
  { id: 'math', title: 'Math Primer', days: [] },
  { id: 'foundations', title: 'Foundations', days: [1, 2] },
  { id: 'classic-ml', title: 'Classic ML', days: [3] },
  { id: 'neural-networks', title: 'Neural Networks', days: [4, 5, 6] },
  { id: 'cnns', title: 'CNNs', days: [7] },
  { id: 'sequences', title: 'Sequences', days: [8] },
  { id: 'transformers', title: 'Transformers', days: [9] },
  { id: 'llms', title: 'Large Language Models', days: [10] },
  { id: 'alignment', title: 'RL & Alignment', days: [11, 12] },
  { id: 'graphs-generative', title: 'Graphs & Generative Models', days: [13] },
  { id: 'applied', title: 'Applied AI', days: [14] },
  { id: 'frontier', title: 'The Frontier', days: [15] },
] as const satisfies readonly { id: string; title: string; days: readonly number[] }[];

export const DAYS = [
  { day: 1, week: 1, title: 'What is learning?' },
  { day: 2, week: 1, title: 'Classification & generalization' },
  { day: 3, week: 1, title: 'Classic machine learning' },
  { day: 4, week: 1, title: 'Neural networks' },
  { day: 5, week: 1, title: 'Backpropagation' },
  { day: 6, week: 1, title: 'Training deep networks' },
  { day: 7, week: 1, title: 'Convolutional neural networks' },
  { day: 8, week: 2, title: 'Sequences & recurrent networks' },
  { day: 9, week: 2, title: 'Transformers' },
  { day: 10, week: 2, title: 'Large language models' },
  { day: 11, week: 2, title: 'Reinforcement learning' },
  { day: 12, week: 2, title: 'Alignment' },
  { day: 13, week: 2, title: 'Graphs & generative models' },
  { day: 14, week: 2, title: 'Applied AI' },
  { day: 15, week: 2, title: 'The frontier (late 2026)' },
] as const;

const byOrder = (a: Lesson, b: Lesson) => a.data.order - b.data.order;

export async function getPath(): Promise<{ day: number; title: string; lessons: Lesson[] }[]> {
  const all = await getCollection('lessons');
  return DAYS.map(({ day, title }) => ({
    day,
    title,
    lessons: all.filter((l) => l.data.day === day).sort(byOrder),
  }));
}

export async function getOrderedLessons(): Promise<Lesson[]> {
  return (await getPath()).flatMap((d) => d.lessons);
}

export async function getMathPrimer(): Promise<Lesson[]> {
  return (await getCollection('lessons')).filter((l) => l.data.track === 'math').sort(byOrder);
}
