/**
 * mask (sft step 4): loss masking on a separate word-level bigram language model (41-word vocabulary, 16 training and 4
 * held-out "user ... assistant ..." transcripts). Phase 1: a cursor sweeps one transcript, token by token, in two rows
 * (every token counted / response tokens only); each counted token gets a bar for its loss under the trained model of that
 * row. Phase 2: the held-out response-token loss of both runs over 300 steps. Both runs go through kit.job (about 100-130 ms
 * each). Props: example (training transcript index, 0). Measured: counted tokens per example 12.6 (every token) vs 5.9
 * (response only); held-out response loss 3.714 at step 0, 2.146 vs 2.150 at step 100, 2.256 vs 2.317 at step 300.
 */
import { TRAIN_TRANSCRIPTS, counted, tokenLosses, type BigramRun } from '../../../lib/align/bigram';
import { int, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { ACCENT, MUTED, RULE, TEXT, circle, clamp, ease, fmt, lineChart, rect, text, type Env } from '../_rl/svg';

const SP = 0.55, CURVES = 6.5, HOLD = 2.5;
interface Scene { all: BigramRun; resp: BigramRun; tokens: string[]; respStart: number; lossAll: Float64Array; lossResp: Float64Array; cAll: number[]; cResp: number[]; avgAll: number; avgResp: number; steps: number }
interface F extends Frame { cursor: number; reveal: number; curveStep: number }

const mode: ModeImpl<Scene, F> = {
  async setup(o: Opts, kit) {
    const [all, resp] = await Promise.all([kit.job<BigramRun>('trainBigram', 'all'), kit.job<BigramRun>('trainBigram', 'response')]);
    const t = TRAIN_TRANSCRIPTS[clamp(int(o.example, 0), 0, TRAIN_TRANSCRIPTS.length - 1)];
    const avg = (m: 'all' | 'response') => TRAIN_TRANSCRIPTS.reduce((s, x) => s + counted(x, m).length, 0) / TRAIN_TRANSCRIPTS.length;
    return {
      all, resp, tokens: t.tokens, respStart: t.respStart, lossAll: tokenLosses(all.W, t), lossResp: tokenLosses(resp.W, t),
      cAll: counted(t, 'all'), cResp: counted(t, 'response'), avgAll: avg('all'), avgResp: avg('response'), steps: all.heldResponse.length - 1,
    };
  },
  timeline: (s) => ({
    duration: s.tokens.length * SP + CURVES + HOLD,
    markers: [{ at: 0, label: 'Which tokens count' }, { at: s.tokens.length * SP, label: 'Held-out loss' }],
  }),
  frameAt(t, s) {
    const n = s.tokens.length, sweepEnd = n * SP, sweeping = t < sweepEnd;
    const cursor = sweeping ? Math.min(n - 1, Math.floor(t / SP + 1e-9)) : -1, reveal = sweeping ? ease((t - cursor * SP) / 0.35) : 1;
    const curveStep = sweeping ? 0 : Math.min(s.steps, Math.floor(clamp((t - sweepEnd) / CURVES, 0, 1) * s.steps + 1e-9));
    const ro: [string, string][] = [
      ['Counted in this example', `${s.cAll.length} vs ${s.cResp.length} tokens`],
      ['Average per training example', `${fmt(s.avgAll, 1)} vs ${fmt(s.avgResp, 1)} tokens`],
    ];
    let status: string;
    if (sweeping) {
      const tok = s.tokens[cursor];
      if (cursor === 0) status = `“${tok}” starts the text. Nothing comes before it to predict it from, so neither run counts it.`;
      else if (cursor < s.respStart) status = `“${tok}” is part of the prompt. The every-token run counts it (loss ${fmt(s.lossAll[cursor], 2)} after training); the response-only run masks it, so its loss on this word is never used.`;
      else status = `“${tok}” is part of the response, so both runs count it: loss ${fmt(s.lossAll[cursor], 2)} when every token is trained on, ${fmt(s.lossResp[cursor], 2)} when only the response is.`;
    } else if (curveStep === 0) status = `Counted tokens in this example: ${s.cAll.length} when every token counts, ${s.cResp.length} when only the response counts. Now both models train from scratch.`;
    else status = `Training step ${curveStep}: held-out response-token loss ${fmt(s.all.heldResponse[curveStep], 3)} with every token counted and ${fmt(s.resp.heldResponse[curveStep], 3)} with response only.`;
    if (!sweeping) ro.push(['Training step', `${curveStep} of ${s.steps}`], ['Held-out loss, every token', fmt(s.all.heldResponse[curveStep], 3)], ['Held-out loss, response only', fmt(s.resp.heldResponse[curveStep], 3)]);
    return { cursor, reveal, curveStep, status, readouts: ro, detail: 'Bars show each token’s loss after 300 training steps of that row’s run.' };
  },
  draw(E: Env, f, s) {
    const nw = E.narrow, W = E.W, H = E.H, pad = nw ? 8 : 12, sz = nw ? 11 : 12, gap = 5, maxW = W - 2 * pad;
    const cw = s.tokens.map((tk) => Math.ceil(tk.length * sz * 0.58) + 10);
    const pos: { x: number; line: number }[] = [];
    let x = 0, line = 0;
    cw.forEach((w) => { if (x > 0 && x + w > maxW) { line++; x = 0; } pos.push({ x: pad + x, line }); x += w + gap; });
    const lines = line + 1, lineH = 58, barMax = 26;
    const rowBlock = 16 + lines * lineH;
    const rows: { label: string; ctd: number[]; loss: Float64Array }[] = [
      { label: 'Trained on every token', ctd: s.cAll, loss: s.lossAll },
      { label: 'Trained on response tokens only', ctd: s.cResp, loss: s.lossResp },
    ];
    rows.forEach((r, ri) => {
      const y0 = 4 + ri * rowBlock;
      text(E, r.label, pad, y0 + 11, { size: 11, fill: MUTED });
      s.tokens.forEach((tk, i) => {
        const p = pos[i], cy = y0 + 16 + p.line * lineH, on = r.ctd.includes(i), cur = f.cursor === i;
        if (on) rect(E, p.x, cy, cw[i], 20, { fill: 'none', stroke: TEXT, sw: 1.25, rx: 4 });
        else rect(E, p.x, cy, cw[i], 20, { fill: RULE, op: 0.55, rx: 4 });
        text(E, tk, p.x + cw[i] / 2, cy + 14, { size: sz, anchor: 'middle', fill: on ? TEXT : MUTED });
        if (cur) rect(E, p.x - 2, cy - 2, cw[i] + 4, 24, { fill: 'none', stroke: ACCENT, sw: 2, rx: 5 });
        const shown = f.cursor < 0 || i < f.cursor ? 1 : i === f.cursor ? f.reveal : 0;
        if (on && shown > 0) {
          const h = Math.min(barMax, (r.loss[i] / 4) * barMax) * shown;
          rect(E, p.x + 2, cy + 24 + barMax - h, cw[i] - 4, h, { fill: 'var(--viz-gradient)', op: 0.9, rx: 2 });
        }
      });
    });
    text(E, 'Grey chips are masked. Bar height: that token’s loss.', W - pad, 4 + 2 * rowBlock + 8, { size: 11, anchor: 'end', fill: MUTED });
    // held-out curves
    const top = 4 + 2 * rowBlock + 34, b = { x: pad + 40, y: top + 14, w: W - 2 * pad - 48, h: H - top - 14 - 44 };
    text(E, 'Loss on the 4 held-out transcripts (response tokens)', pad, top, { size: 11, fill: MUTED });
    const ax = { lo: 0, hi: s.steps, ticks: [0, 100, 200, 300].filter((v) => v <= s.steps), label: 'Training step' }, ay = { lo: 1.8, hi: 3.8, ticks: [2, 2.5, 3, 3.5] };
    const k = f.curveStep, ysR = s.resp.heldResponse, ysA = s.all.heldResponse;
    const { sx, sy } = lineChart(E, b, ax, ay, [{ ys: ysR, color: 'var(--viz-forward)', width: 3.5 }, { ys: ysA, color: TEXT, width: 1.5 }], k);
    { circle(E, sx(k), sy(ysR[k]), 4, { fill: 'var(--viz-forward)' }); circle(E, sx(k), sy(ysA[k]), 2.5, { fill: TEXT }); }
  },
  label: (s) => `Loss masking on a tiny word model: one transcript with ${s.cAll.length} tokens counted when every token trains and ${s.cResp.length} when only the response does, and the held-out response loss of both runs, which end at ${fmt(s.all.heldResponse[s.steps], 2)} and ${fmt(s.resp.heldResponse[s.steps], 2)}.`,
};
export default mode;
