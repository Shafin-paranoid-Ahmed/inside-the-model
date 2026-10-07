import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'Six models, one 200-question test',
  legend: [
    { label: 'Answered right', color: '--viz-pos' },
    { label: 'Answered wrong', color: '--viz-neg' },
    { label: 'Score with 95% interval', color: '--text-muted', shape: 'line' },
    { label: 'True score', color: '--text', shape: 'line' },
  ],
  aspect: '720 / 400',
  narrowAspect: '360 / 520',
  lanes: 2,
} satisfies ModeMeta;
