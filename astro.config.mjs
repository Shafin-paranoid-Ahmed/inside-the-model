// @ts-check
import { defineConfig } from 'astro/config';
import { fontProviders } from 'astro/config';
import svelte from '@astrojs/svelte';
import mdx from '@astrojs/mdx';
import { unified } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

import sitemap from '@astrojs/sitemap';

// Sätteri is Astro 7's default processor but has no remark/rehype support;
// unified() is used for build-time KaTeX. Applies to .md and .mdx.
// https://astro.build/config
export default defineConfig({
  // TODO: replace with the real domain before launch
  site: 'https://inside-model.vercel.app',
  fonts: [
    { provider: fontProviders.local(), name: 'Source Serif 4', cssVariable: '--font-serif', fallbacks: ['Georgia', 'serif'], options: { variants: [
      { weight: '200 900', style: 'normal', src: ['./node_modules/@fontsource-variable/source-serif-4/files/source-serif-4-latin-wght-normal.woff2'] },
      { weight: '200 900', style: 'italic', src: ['./node_modules/@fontsource-variable/source-serif-4/files/source-serif-4-latin-wght-italic.woff2'] } ] } },
    { provider: fontProviders.local(), name: 'Inter', cssVariable: '--font-sans', fallbacks: ['sans-serif'], options: { variants: [
      { weight: '100 900', style: 'normal', src: ['./node_modules/@fontsource-variable/inter/files/inter-latin-wght-normal.woff2'] } ] } },
    { provider: fontProviders.local(), name: 'JetBrains Mono', cssVariable: '--font-mono', fallbacks: ['monospace'], options: { variants: [
      { weight: '100 800', style: 'normal', src: ['./node_modules/@fontsource-variable/jetbrains-mono/files/jetbrains-mono-latin-wght-normal.woff2'] } ] } },
  ],
  integrations: [svelte(), mdx(), sitemap()],
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [rehypeKatex],
    }),
  },
});