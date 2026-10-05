import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'The Q-learning target for three kinds of move',
  legend: [
    { label: 'Card being built', color: '--accent', shape: 'line' },
    { label: 'Positive value', color: '--viz-pos' },
    { label: 'Negative value', color: '--viz-neg' },
  ],
  aspect: '720 / 290',
  lanes: 1,
  narrowAspect: '360 / 640',
} satisfies ModeMeta;
