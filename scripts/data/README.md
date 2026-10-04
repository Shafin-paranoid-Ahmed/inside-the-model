# scripts/data

## alice.txt

Lewis Carroll, *Alice's Adventures in Wonderland* and *Through the Looking-Glass*, both public domain, from Project Gutenberg:

- #11: https://www.gutenberg.org/cache/epub/11/pg11.txt
- #12: https://www.gutenberg.org/cache/epub/12/pg12.txt

Retrieved 2026-10-04 with `curl`. Processing (`prepareAlice` in `scripts/train-models.ts`):

1. Keep the text between the `*** START OF THE PROJECT GUTENBERG EBOOK ... ***` and `*** END OF ... ***` lines (Gutenberg header, licence and footer removed).
2. Drop `[Illustration]` markers; lowercase; map curly quotes to `'` and `"`, em and en dashes to `-`, `æ` to `ae`; remove `_` (italics).
3. Join each paragraph's hard-wrapped lines into one line, keep only characters in `CHARSET` (`src/lib/lm/charset.ts`: a–z, space, newline, `.,;:'!?-"`), collapse repeated spaces, and drop empty paragraphs. Paragraphs are separated by one newline; book 11 comes first.

Result: 303,013 characters. Rebuild with:

```bash
npx esbuild scripts/train-models.ts --bundle --platform=node --format=esm --outfile="$TEMP/train-models.mjs"
node "$TEMP/train-models.mjs" corpus pg11.txt pg12.txt
```

The Day 10 jobs split it into 50 contiguous chunks and hold out every 10th (indexes 9, 19, 29, 39, 49) for validation, a 90/10 split.

The facts model's people are invented by `factsWorld()` in the same script from seeded syllables (seed 1212); no real people and no data file.
