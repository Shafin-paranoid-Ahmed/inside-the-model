import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'Fine-tuning copies the demonstrators',
  legend: [
    { label: 'Correct answers after fine-tuning', color: '--viz-forward', shape: 'line' },
    { label: 'Share of demonstrations that are correct', color: '--text', shape: 'line' },
  ],
  aspect: '720 / 420',
  narrowAspect: '360 / 520',
  lanes: 1,
} satisfies ModeMeta;
