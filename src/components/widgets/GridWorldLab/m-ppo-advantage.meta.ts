import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'Advantage: how much better than the critic expected',
  legend: [
    { label: 'Positive error', color: '--viz-pos' },
    { label: 'Negative error', color: '--viz-neg' },
    { label: 'Share of step 1’s advantage', color: '--viz-gradient', shape: 'line' },
  ],
  aspect: '720 / 400',
  narrowAspect: '360 / 600',
  lanes: 2,
} satisfies ModeMeta;
