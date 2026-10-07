import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'Demonstrations written by people',
  legend: [
    { label: 'Demonstration is correct', color: '--viz-pos' },
    { label: 'Demonstration is wrong', color: '--viz-neg' },
    { label: 'Share or length so far', color: '--viz-forward' },
  ],
  aspect: '720 / 470',
  narrowAspect: '360 / 680',
  lanes: 1,
} satisfies ModeMeta;
