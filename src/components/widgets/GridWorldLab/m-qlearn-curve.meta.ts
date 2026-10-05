import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'How much to explore: three ε on the same world',
  legend: [
    { label: 'ε = 0 (never explore)', color: '--viz-cat-1', shape: 'line' },
    { label: 'ε = 0.1', color: '--viz-cat-7', shape: 'line' },
    { label: 'ε = 0.3', color: '--viz-cat-3', shape: 'line' },
    { label: 'Greedy action is optimal', color: '--text', shape: 'line' },
    { label: 'Greedy action is not optimal', color: '--viz-neg', shape: 'line' },
  ],
  aspect: '720 / 480',
  lanes: 2,
  narrowAspect: '360 / 700',
  loading: 'Training 60 Q-learning runs in your browser…',
} satisfies ModeMeta;
