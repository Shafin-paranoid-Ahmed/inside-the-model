import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'One PPO update on a batch of responses',
  legend: [
    { label: 'Positive advantage', color: '--viz-pos', shape: 'dot' },
    { label: 'Negative advantage', color: '--viz-neg', shape: 'dot' },
    { label: 'Hollow dot: flat, no gradient', color: '--text-muted', shape: 'dot' },
    { label: 'Clip band', color: '--accent' },
  ],
  aspect: '720 / 470',
  narrowAspect: '360 / 600',
  lanes: 2,
  pickable: true,
  loading: 'Running the update in your browser…',
} satisfies ModeMeta;
