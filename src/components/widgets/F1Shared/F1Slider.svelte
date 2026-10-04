<script lang="ts">
  interface Props {
    label: string;
    value: number;
    min: number;
    max: number;
    step: number;
    /** Decimals shown in the readout. */
    digits?: number;
    /** Called with the new value on user input. */
    oninput: (v: number) => void;
  }
  let { label, value, min, max, step, digits = 2, oninput }: Props = $props();
</script>

<label class="sl">
  <span class="nm">{label}</span>
  <input type="range" {min} {max} {step} {value} oninput={(e) => oninput(+e.currentTarget.value)} />
  <output class="num">{value.toFixed(digits).replace('-', '−')}</output>
</label>

<style>
  .sl { display: inline-flex; align-items: center; gap: var(--s-2); color: var(--text-muted); }
  .nm { min-width: 1.2rem; }
  input { width: 8rem; accent-color: var(--accent); }
  output { min-width: 3.4rem; color: var(--text); font: 0.85rem var(--font-code); font-variant-numeric: tabular-nums; }
</style>
