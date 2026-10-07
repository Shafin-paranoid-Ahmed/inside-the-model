import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'What a base model does with a question',
  legend: [
    { label: 'Probability of the response', color: '--viz-forward' },
    { label: 'Continues the question list', color: '--accent' },
  ],
  aspect: '720 / 460',
  narrowAspect: '360 / 600',
  lanes: 3,
} satisfies ModeMeta;
