import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'Reward against distance from the SFT model',
  legend: [
    { label: 'Proxy reward (reward model)', color: '--viz-forward', shape: 'line' },
    { label: 'Gold reward', color: '--text', shape: 'line' },
    { label: 'Gap between them', color: '--text-muted' },
    { label: 'Gold peak', color: '--accent', shape: 'line' },
  ],
  aspect: '720 / 430',
  narrowAspect: '360 / 480',
  lanes: 2,
  loading: 'Training the policy in your browser…',
} satisfies ModeMeta;
