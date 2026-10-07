import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'Edit the constitution, retrain on AI feedback',
  legend: [
    { label: 'Chance of the response (average over prompts)', color: '--viz-forward' },
    { label: 'Before training', color: '--text-muted', shape: 'line' },
    { label: 'Default constitution, trained', color: '--text', shape: 'line' },
    { label: 'Gold reward', color: '--text', shape: 'line' },
  ],
  aspect: '720 / 420',
  narrowAspect: '360 / 600',
  lanes: 2,
  loading: 'Training the policy on AI feedback…',
  controls: [
    { key: 'answer', label: 'Answer the question', kind: 'toggle' },
    { key: 'accurate', label: 'Be accurate', kind: 'toggle' },
    { key: 'flattery', label: 'No flattery', kind: 'toggle' },
    { key: 'refuse', label: 'No needless refusal', kind: 'toggle' },
    { key: 'cautious', label: 'Avoid anything that could be wrong', kind: 'toggle' },
  ],
} satisfies ModeMeta;
