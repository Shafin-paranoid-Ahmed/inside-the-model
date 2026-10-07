import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'Eight attempts at each problem',
  legend: [
    { label: 'Pass', color: '--viz-pos' },
    { label: 'Fail', color: '--viz-neg' },
    { label: 'Bar: lines of working written', color: '--text-muted' },
  ],
  aspect: '720 / 340',
  narrowAspect: '360 / 520',
  lanes: 2,
} satisfies ModeMeta;
