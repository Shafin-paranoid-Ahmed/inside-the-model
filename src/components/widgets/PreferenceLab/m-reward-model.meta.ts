import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'Training a reward model on the pairs',
  legend: [
    { label: 'Learned weight above 0', color: '--viz-pos' },
    { label: 'Learned weight below 0', color: '--viz-neg' },
    { label: 'Gold weight', color: '--text', shape: 'line' },
    { label: 'Held-out accuracy', color: '--viz-forward', shape: 'line' },
    { label: 'Training accuracy', color: '--text-muted', shape: 'line' },
  ],
  aspect: '720 / 440',
  narrowAspect: '360 / 640',
  lanes: 2,
} satisfies ModeMeta;
