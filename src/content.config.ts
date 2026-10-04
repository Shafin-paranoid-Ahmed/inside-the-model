import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

export const TRACK_IDS = [
  'math',
  'foundations',
  'classic-ml',
  'neural-networks',
  'cnns',
  'sequences',
  'transformers',
  'llms',
  'alignment',
  'graphs-generative',
  'applied',
  'frontier',
] as const;

// Entry id = path relative to base without extension: `<track>/<slug>`.
const lessons = defineCollection({
  loader: glob({ base: './src/content/lessons', pattern: '**/*.mdx' }),
  schema: z
    .object({
      title: z.string(),
      summary: z.string(),
      day: z.number().int().min(1).max(15).nullable(),
      track: z.enum(TRACK_IDS),
      order: z.number(),
      level: z.enum(['intro', 'core', 'advanced', 'frontier']),
      duration: z.number(),
      prereqs: z.array(z.string()).default([]),
      status: z.enum(['stub', 'draft', 'done']).default('stub'),
      lastReviewed: z.coerce.date().optional(),
    })
    .refine((d) => d.level !== 'frontier' || d.lastReviewed, {
      message: 'frontier lessons require lastReviewed',
      path: ['lastReviewed'],
    }),
});

export const collections = { lessons };
