import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'Old probabilities stay fixed, new ones move',
  legend: [
    { label: 'Old probability (saved)', color: '--text-muted', shape: 'line' },
    { label: 'New probability', color: '--viz-forward' },
    { label: 'Selected sample', color: '--accent' },
  ],
  aspect: '720 / 400',
  narrowAspect: '360 / 600',
  lanes: 2,
  pickable: true,
} satisfies ModeMeta;
