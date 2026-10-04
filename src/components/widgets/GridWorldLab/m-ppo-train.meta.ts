import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'PPO training on the grid: collect, update, repeat',
  legend: [
    { label: 'Policy probabilities', color: '--viz-forward', shape: 'line' },
    { label: 'Value of the start cell', color: '--viz-pos', shape: 'line' },
    { label: 'Best possible', color: '--text-muted', shape: 'line' },
    { label: 'Critic error', color: '--viz-neg', shape: 'line' },
  ],
  aspect: '720 / 420',
  narrowAspect: '360 / 660',
  lanes: 2,
  loading: 'Training PPO for 100 iterations in your browser…',
} satisfies ModeMeta;
