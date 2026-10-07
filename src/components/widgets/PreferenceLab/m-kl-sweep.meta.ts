import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'Four penalty strengths, same reward model',
  legend: [
    { label: 'Path of one run', color: '--text-muted', shape: 'line' },
    { label: 'Where it is now', color: '--text' },
    { label: 'Best gold reward so far', color: '--accent' },
  ],
  aspect: '720 / 430',
  narrowAspect: '360 / 480',
  lanes: 2,
  loading: 'Training four policies in your browser…',
} satisfies ModeMeta;
