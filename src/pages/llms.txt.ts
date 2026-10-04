import type { APIRoute } from 'astro';
import { getPath, getMathPrimer } from '../lib/curriculum';
import { absUrl } from '../lib/seo';

export const GET: APIRoute = async () => {
  const line = (l: { id: string; data: { title: string; summary: string } }) => `- [${l.data.title}](${absUrl(`/learn/${l.id}`)}): ${l.data.summary}`;
  const days = await getPath();
  const math = await getMathPrimer();
  const body = [
    '# AI Learner',
    '',
    '> A free, no-login course that teaches how AI actually works in two weeks, through interactive visualizations of real mechanisms.',
    '',
    'Fifteen days from "what is a model?" to the frontier: backpropagation, sequence models, transformers, LLMs, alignment and generative models. Every lesson is a few short steps built around a small model running in your browser. Nothing leaves your device.',
    '',
    `Full path: ${absUrl('/path')}`,
    '',
    '## Curriculum',
    '',
    ...days.flatMap((d) => [`### Day ${d.day}: ${d.title}`, '', ...d.lessons.map(line), '']),
    '## Math primer',
    '',
    ...math.map(line),
    '',
  ].join('\n');
  return new Response(body, { headers: { 'Content-Type': 'text/plain; charset=utf-8' } });
};
