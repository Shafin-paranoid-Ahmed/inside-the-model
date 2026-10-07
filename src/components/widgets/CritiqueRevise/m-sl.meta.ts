import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'Fine-tuning on the revised responses',
  legend: [
    { label: 'Chance of the response (average over prompts)', color: '--viz-forward' },
    { label: 'Before fine-tuning', color: '--text-muted', shape: 'line' },
    { label: 'Terse wrong answer', color: '--accent' },
    { label: 'Principles broken per response', color: '--viz-neg', shape: 'line' },
  ],
  aspect: '720 / 400',
  narrowAspect: '360 / 560',
  lanes: 2,
} satisfies ModeMeta;
