import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'What each action is worth, cell by cell',
  legend: [
    { label: 'Positive Q', color: '--viz-pos' },
    { label: 'Negative Q', color: '--viz-neg' },
    { label: 'Best action (outlined)', color: '--accent', shape: 'line' },
  ],
  aspect: '720 / 400',
  lanes: 2,
  narrowAspect: '360 / 540',
} satisfies ModeMeta;
