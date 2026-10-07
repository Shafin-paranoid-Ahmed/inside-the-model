import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'Labelers compare two answers',
  legend: [
    { label: 'Agrees with gold', color: '--viz-pos' },
    { label: 'Disagrees with gold', color: '--viz-neg' },
    { label: 'Chosen / shown pair', color: '--accent' },
  ],
  aspect: '720 / 250',
  narrowAspect: '360 / 440',
  pickable: true,
} satisfies ModeMeta;
