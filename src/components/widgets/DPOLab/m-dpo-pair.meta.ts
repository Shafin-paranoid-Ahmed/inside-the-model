import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'DPO on one preference pair',
  legend: [
    { label: 'Policy probability', color: '--viz-forward' },
    { label: 'Reference probability', color: '--text-muted' },
    { label: 'Gradient weight', color: '--viz-gradient', shape: 'line' },
  ],
  aspect: '720 / 320',
  lanes: 2,
  narrowAspect: '360 / 600',
} satisfies ModeMeta;
