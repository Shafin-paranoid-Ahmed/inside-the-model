import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'One network answers for every cell',
  legend: [
    { label: 'Positive Q or contribution', color: '--viz-pos' },
    { label: 'Negative Q or contribution', color: '--viz-neg' },
    { label: 'Active hidden unit', color: '--viz-seq-4' },
    { label: 'Selected cell', color: '--accent' },
  ],
  aspect: '720 / 400',
  narrowAspect: '360 / 640',
  pickable: true,
} satisfies ModeMeta;
