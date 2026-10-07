import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'Training on the demonstrations',
  legend: [
    { label: 'Chance of the response now', color: '--viz-forward' },
    { label: 'Base model (outline and tick)', color: '--text-muted' },
    { label: 'Weight up / down', color: '--viz-pos' },
  ],
  aspect: '720 / 480',
  narrowAspect: '360 / 720',
  lanes: 1,
} satisfies ModeMeta;
