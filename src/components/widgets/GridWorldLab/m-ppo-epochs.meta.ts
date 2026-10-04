import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'Four passes over one frozen batch',
  legend: [
    { label: 'Positive advantage', color: '--viz-pos' },
    { label: 'Negative advantage', color: '--viz-neg' },
    { label: 'Still pushing', color: '--viz-forward' },
    { label: 'Flat (no push)', color: '--text-muted' },
    { label: 'Clip band', color: '--accent' },
  ],
  aspect: '720 / 420',
  narrowAspect: '360 / 640',
  lanes: 2,
} satisfies ModeMeta;
