import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'Reward model score against gold quality',
  legend: [
    { label: 'Continues list', color: '--viz-cat-1' },
    { label: 'Terse correct', color: '--viz-cat-2' },
    { label: 'Clear correct', color: '--viz-cat-3' },
    { label: 'Padded correct', color: '--viz-cat-4' },
    { label: 'Rambling wrong', color: '--viz-cat-5' },
    { label: 'Terse wrong', color: '--viz-cat-6' },
    { label: 'Refusal', color: '--viz-cat-7' },
    { label: 'Hedged correct', color: '--viz-cat-8' },
    { label: 'Trend of the other seven', color: '--text-muted', shape: 'line' },
  ],
  aspect: '720 / 440',
  narrowAspect: '360 / 500',
  lanes: 2,
} satisfies ModeMeta;
