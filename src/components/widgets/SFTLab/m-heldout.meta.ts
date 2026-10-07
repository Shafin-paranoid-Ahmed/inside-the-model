import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'Prompts the model never saw in training',
  legend: [
    { label: 'Chance of the response now', color: '--viz-forward' },
    { label: 'Base model (outline)', color: '--text-muted' },
  ],
  aspect: '720 / 420',
  narrowAspect: '360 / 560',
  lanes: 1,
} satisfies ModeMeta;
