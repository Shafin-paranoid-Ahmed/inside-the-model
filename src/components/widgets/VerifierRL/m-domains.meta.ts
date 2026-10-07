import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'Prompts with a checker and prompts without',
  legend: [
    { label: 'Policy probability', color: '--viz-forward' },
    { label: 'Dashed outline: where training started', color: '--text-muted' },
  ],
  aspect: '720 / 460',
  narrowAspect: '360 / 760',
  loading: 'Training 150 iterations in your browser…',
} satisfies ModeMeta;
