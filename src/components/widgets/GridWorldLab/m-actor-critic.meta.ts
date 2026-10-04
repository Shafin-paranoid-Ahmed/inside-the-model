import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'Actor-critic: a critic scores every step, the actor follows',
  legend: [
    { label: 'Critic value V (cell colour and number)', color: '--viz-pos' },
    { label: 'Actor policy (arrows)', color: '--text', shape: 'line' },
    { label: 'REINFORCE, no baseline', color: '--viz-cat-1', shape: 'line' },
    { label: 'REINFORCE, learned baseline', color: '--viz-cat-2', shape: 'line' },
    { label: 'Actor-critic', color: '--viz-cat-3', shape: 'line' },
  ],
  aspect: '720 / 460',
  narrowAspect: '360 / 680',
  lanes: 2,
  loading: 'Training 30 runs in your browser…',
} satisfies ModeMeta;
