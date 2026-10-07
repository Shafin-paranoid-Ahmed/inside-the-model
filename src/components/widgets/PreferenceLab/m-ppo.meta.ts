import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'PPO against the reward model',
  legend: [
    { label: 'Proxy reward (reward model)', color: '--viz-forward', shape: 'line' },
    { label: 'Gold reward', color: '--text', shape: 'line' },
    { label: 'KL to the SFT model', color: '--text-muted', shape: 'line' },
    { label: 'Sampled, above its prompt average', color: '--viz-pos' },
    { label: 'Sampled, below it', color: '--viz-neg' },
  ],
  aspect: '720 / 470',
  narrowAspect: '360 / 610',
  lanes: 3,
  loading: 'Training the policy in your browser…',
} satisfies ModeMeta;
