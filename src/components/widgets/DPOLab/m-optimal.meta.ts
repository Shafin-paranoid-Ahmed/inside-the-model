import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'The best policy for a given penalty strength',
  legend: [
    { label: 'π* (the optimum)', color: '--viz-forward' },
    { label: 'Reference π_ref', color: '--text-muted' },
    { label: 'PPO end state', color: '--text', shape: 'line' },
    { label: 'Reward model score', color: '--viz-forward', shape: 'line' },
    { label: 'Gold reward', color: '--text', shape: 'line' },
  ],
  aspect: '720 / 400',
  narrowAspect: '360 / 660',
  loading: 'Running PPO at β = 0.1 for the tick marks…',
} satisfies ModeMeta;
