import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'Which tokens count toward the loss',
  legend: [
    { label: 'Loss of a counted token (bar)', color: '--viz-gradient' },
    { label: 'Held-out loss, every token counted', color: '--text', shape: 'line' },
    { label: 'Held-out loss, response only', color: '--viz-forward', shape: 'line' },
  ],
  aspect: '720 / 450',
  narrowAspect: '360 / 700',
  lanes: 1,
  loading: 'Training two small language models in your browser…',
} satisfies ModeMeta;
