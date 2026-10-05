import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'Values of a fixed policy, one sweep at a time',
  legend: [
    { label: 'Positive value', color: '--viz-pos' },
    { label: 'Negative value', color: '--viz-neg' },
    { label: 'Where the move can land', color: '--viz-forward', shape: 'line' },
  ],
  aspect: '720 / 380',
  lanes: 2,
  narrowAspect: '360 / 520',
} satisfies ModeMeta;
