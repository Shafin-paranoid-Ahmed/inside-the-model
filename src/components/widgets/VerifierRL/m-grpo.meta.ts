import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'Scoring each attempt against its group',
  legend: [
    { label: 'Above the group average', color: '--viz-pos' },
    { label: 'Below the group average', color: '--viz-neg' },
    { label: 'Policy probability', color: '--viz-forward' },
  ],
  aspect: '720 / 380',
  narrowAspect: '360 / 700',
  lanes: 3,
} satisfies ModeMeta;
