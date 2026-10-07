import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'DPO trained on all 200 pairs',
  legend: [
    { label: 'Policy probability', color: '--viz-forward' },
    { label: 'Reference', color: '--text-muted' },
    { label: 'π* (the optimum)', color: '--text', shape: 'line' },
    { label: 'Reward model score', color: '--viz-forward', shape: 'line' },
    { label: 'Gold reward', color: '--text', shape: 'line' },
  ],
  aspect: '720 / 440',
  narrowAspect: '360 / 720',
  lanes: 2,
  loading: 'Training DPO on 200 pairs…',
} satisfies ModeMeta;
