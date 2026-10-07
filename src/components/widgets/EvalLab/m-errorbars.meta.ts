import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'How often does the weaker model come out ahead?',
  legend: [
    { label: 'Each model gets its own questions', color: '--viz-cat-3', shape: 'line' },
    { label: 'Both models get the same questions', color: '--viz-cat-7', shape: 'line' },
    { label: '5% line', color: '--text', shape: 'line' },
  ],
  aspect: '720 / 400',
  narrowAspect: '360 / 560',
  lanes: 2,
} satisfies ModeMeta;
