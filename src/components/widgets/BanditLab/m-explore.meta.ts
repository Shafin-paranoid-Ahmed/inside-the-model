import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'Greedy, ε-greedy and UCB on the same bandits',
  legend: [
    { label: 'Estimated mean', color: '--viz-forward' },
    { label: 'True mean', color: '--text', shape: 'line' },
    { label: 'Greedy', color: '--viz-cat-1', shape: 'line' },
    { label: 'ε = 0.1', color: '--viz-cat-2', shape: 'line' },
    { label: 'UCB', color: '--viz-cat-3', shape: 'line' },
  ],
  aspect: '720 / 400',
  lanes: 2,
  narrowAspect: '360 / 640',
} satisfies ModeMeta;
