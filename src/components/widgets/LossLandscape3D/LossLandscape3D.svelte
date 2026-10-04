<script lang="ts">
  import { onMount } from 'svelte';
  import {
    EXTENT, H, TRAIL, OPTS, loss, makeBall, stepBall, startPoint, buildMesh, viewProj, project, rayCast, eye,
    type Ball, type Vec3,
  } from './sim';

  const N = 112, STEPS_PER_S = 90, REVEAL_MS = 1300, HOLD_MS = 1600, FADE_MS = 700;
  const NAMES = OPTS;

  let canvas: HTMLCanvasElement, root: HTMLElement;
  let tags: HTMLElement[] = $state([]);
  let axes: HTMLElement[] = $state([]);
  let failed = $state(false);
  let live = $state('');
  let drop = () => {};

  onMount(() => {
    const gl = canvas.getContext('webgl2', { antialias: true, alpha: false });
    if (!gl) { failed = true; return; }
    const html = document.documentElement;
    const reduced = matchMedia('(prefers-reduced-motion: reduce)');

    // ---- shaders ----------------------------------------------------------------------------
    const SURF_VS = `#version 300 es
      in vec3 aPos; in vec3 aNrm; in float aT;
      uniform mat4 uVP;
      out vec3 vPos; out vec3 vNrm; out float vT;
      void main() { vPos = aPos; vNrm = aNrm; vT = aT; gl_Position = uVP * vec4(aPos, 1.0); }`;
    const SURF_FS = `#version 300 es
      precision highp float;
      in vec3 vPos; in vec3 vNrm; in float vT;
      uniform vec3 uC[5]; uniform vec3 uBg; uniform vec3 uLine; uniform vec3 uEye; uniform float uReveal;
      out vec4 o;
      vec3 ramp(float t) {
        t = mix(0.8, 4.0, clamp(1.0 - t, 0.0, 1.0));
        int i = int(min(floor(t), 3.0));
        return mix(uC[i], uC[i + 1], t - float(i));
      }
      float lineAA(float v) { float w = fwidth(v); return 1.0 - smoothstep(0.0, w * 1.2, abs(fract(v - 0.5) - 0.5)); }
      void main() {
        float edge = uReveal * 1.2 - length(vPos.xz) / ${(EXTENT * Math.SQRT2).toFixed(4)};
        if (edge < 0.0) discard;
        vec3 n = normalize(vNrm);
        float diff = max(dot(n, normalize(vec3(-0.45, 1.0, 0.35))), 0.0);
        vec3 col = ramp(vT) * (0.6 + 0.48 * diff);
        col = mix(col, uLine, lineAA(vT * 16.0) * 0.32);
        col = mix(col, uLine, max(lineAA(vPos.x * 2.0), lineAA(vPos.z * 2.0)) * 0.07);
        col = mix(col, uBg, smoothstep(10.5, 17.0, distance(uEye, vPos)) * 0.6);
        col = mix(col, uLine, (1.0 - smoothstep(0.0, 0.05, edge)) * 0.7);
        o = vec4(col, 1.0);
      }`;
    const LINE_VS = `#version 300 es
      in vec3 aPos; in float aA;
      uniform mat4 uVP; uniform float uSize;
      out float vA;
      void main() { vA = aA; gl_Position = uVP * vec4(aPos, 1.0); gl_PointSize = uSize; }`;
    const LINE_FS = `#version 300 es
      precision highp float;
      in float vA;
      uniform vec3 uCol; uniform vec3 uRim; uniform int uPoint;
      out vec4 o;
      void main() {
        float a = vA;
        vec3 c = uCol;
        if (uPoint == 1) {
          float r = length(gl_PointCoord * 2.0 - 1.0);
          if (r > 1.0) discard;
          c = mix(uCol, uRim, smoothstep(0.6, 0.72, r));
          a *= 1.0 - smoothstep(0.88, 1.0, r);
        }
        o = vec4(c * a, a);
      }`;

    function program(vs: string, fs: string) {
      const p = gl!.createProgram()!;
      for (const [type, src] of [[gl!.VERTEX_SHADER, vs], [gl!.FRAGMENT_SHADER, fs]] as const) {
        const s = gl!.createShader(type)!;
        gl!.shaderSource(s, src);
        gl!.compileShader(s);
        if (!gl!.getShaderParameter(s, gl!.COMPILE_STATUS)) throw new Error(gl!.getShaderInfoLog(s) ?? 'shader');
        gl!.attachShader(p, s);
      }
      gl!.linkProgram(p);
      if (!gl!.getProgramParameter(p, gl!.LINK_STATUS)) throw new Error(gl!.getProgramInfoLog(p) ?? 'link');
      return p;
    }
    let surf: WebGLProgram, line: WebGLProgram;
    try { surf = program(SURF_VS, SURF_FS); line = program(LINE_VS, LINE_FS); }
    catch { failed = true; return; }
    const U = (p: WebGLProgram, n: string) => gl.getUniformLocation(p, n);

    // ---- surface geometry ------------------------------------------------------------------
    const mesh = buildMesh(N);
    const surfVao = gl.createVertexArray()!;
    gl.bindVertexArray(surfVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, mesh.verts, gl.STATIC_DRAW);
    const attr = (p: WebGLProgram, name: string, size: number, stride: number, off: number) => {
      const loc = gl.getAttribLocation(p, name);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, stride * 4, off * 4);
    };
    attr(surf, 'aPos', 3, 7, 0);
    attr(surf, 'aNrm', 3, 7, 3);
    attr(surf, 'aT', 1, 7, 6);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, mesh.idx, gl.STATIC_DRAW);

    // dynamic lines/points: [x, y, z, alpha]
    const lineVao = gl.createVertexArray()!;
    gl.bindVertexArray(lineVao);
    const lineBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, lineBuf);
    attr(line, 'aPos', 3, 4, 0);
    attr(line, 'aA', 1, 4, 3);
    gl.bindVertexArray(null);
    const scratch = new Float32Array(4 * (TRAIL + 8) * 3);

    // axis labels sit just past the middle of two surface edges
    const E = EXTENT + 0.35;
    const AXES: Vec3[] = [[0, H * loss(0, EXTENT) + 0.1, E], [E, H * loss(EXTENT, 0) + 0.1, 0]];

    // ---- colors from tokens ----------------------------------------------------------------
    const hex = (n: string): Vec3 => {
      const h = getComputedStyle(html).getPropertyValue(n).trim().replace('#', '');
      return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16) / 255) as Vec3;
    };
    let col = read();
    function read() {
      return {
        bg: hex('--bg'), line: hex('--text'),
        seq: [1, 2, 3, 4, 5].map((i) => hex(`--viz-seq-${i}`)),
        balls: [hex('--viz-neg'), hex('--viz-forward'), hex('--accent')],
      };
    }

    // ---- simulation state ------------------------------------------------------------------
    let run = 0, userDrop = false;
    let balls: Ball[] = [];
    let phase: 'run' | 'hold' | 'fade' = 'run';
    let phaseT = 0, alpha = 0, stepAcc = 0;
    function start(x: number, y: number) {
      balls = NAMES.map((n) => makeBall(n, x, y));
      phase = 'run'; phaseT = 0; stepAcc = 0;
      if (reduced.matches) { balls.forEach((b) => { while (!stepBall(b)); }); phase = 'hold'; alpha = 1; }
    }
    function announce() {
      const r = balls.map((b) => `${b.name} ${loss(b.x, b.y).toFixed(2)}`).join(', ');
      live = `Final loss: ${r}. Lower is better.`;
    }
    start(...startPoint(run));

    // ---- camera + input --------------------------------------------------------------------
    let yaw = -0.55, pitch = 0.5, yawVel = 0, pitchVel = 0, dragging = false, moved = 0;
    let px = 0, py = 0, aspect = 1, vp = viewProj(yaw, pitch, aspect);
    const AUTO = 0.07;
    const onDown = (e: PointerEvent) => {
      dragging = true; moved = 0; px = e.clientX; py = e.clientY;
      canvas.setPointerCapture(e.pointerId);
    };
    const onMove = (e: PointerEvent) => {
      if (!dragging) return;
      const dx = e.clientX - px, dy = e.clientY - py;
      px = e.clientX; py = e.clientY; moved += Math.abs(dx) + Math.abs(dy);
      yawVel = -dx * 0.006; pitchVel = dy * 0.004;
      yaw += yawVel; pitch = Math.min(1.1, Math.max(0.22, pitch + pitchVel));
      wake();
    };
    const onUp = (e: PointerEvent) => {
      dragging = false;
      if (moved < 6) {
        const r = canvas.getBoundingClientRect();
        const hit = rayCast(yaw, pitch, aspect, ((e.clientX - r.left) / r.width) * 2 - 1, 1 - ((e.clientY - r.top) / r.height) * 2);
        if (hit) { start(hit[0], hit[1]); alpha = 1; userDrop = true; wake(); }
      }
    };
    canvas.addEventListener('pointerdown', onDown);
    canvas.addEventListener('pointermove', onMove);
    canvas.addEventListener('pointerup', onUp);
    canvas.addEventListener('pointercancel', () => (dragging = false));
    drop = () => { run++; start(...startPoint(run)); alpha = 1; userDrop = true; wake(); };

    // ---- drawing ---------------------------------------------------------------------------
    function drawLines(data: Float32Array, n: number, mode: number, rgb: Vec3, size = 1) {
      gl!.useProgram(line);
      gl!.bindVertexArray(lineVao);
      gl!.bindBuffer(gl!.ARRAY_BUFFER, lineBuf);
      gl!.bufferData(gl!.ARRAY_BUFFER, data.subarray(0, n * 4), gl!.DYNAMIC_DRAW);
      gl!.uniformMatrix4fv(U(line, 'uVP'), false, vp);
      gl!.uniform3fv(U(line, 'uCol'), rgb);
      gl!.uniform3fv(U(line, 'uRim'), col.bg);
      gl!.uniform1f(U(line, 'uSize'), size);
      gl!.uniform1i(U(line, 'uPoint'), mode === gl!.POINTS ? 1 : 0);
      gl!.drawArrays(mode, 0, n);
    }

    let reveal = reduced.matches ? 1 : 0;
    function render() {
      const d = Math.min(2, devicePixelRatio || 1);
      gl!.viewport(0, 0, canvas.width, canvas.height);
      gl!.clearColor(col.bg[0], col.bg[1], col.bg[2], 1);
      gl!.clear(gl!.COLOR_BUFFER_BIT | gl!.DEPTH_BUFFER_BIT);
      gl!.enable(gl!.DEPTH_TEST);
      vp = viewProj(yaw, pitch, aspect);

      gl!.useProgram(surf);
      gl!.bindVertexArray(surfVao);
      gl!.uniformMatrix4fv(U(surf, 'uVP'), false, vp);
      gl!.uniform3fv(U(surf, 'uC'), col.seq.flat());
      gl!.uniform3fv(U(surf, 'uBg'), col.bg);
      gl!.uniform3fv(U(surf, 'uLine'), col.line);
      gl!.uniform3fv(U(surf, 'uEye'), eye(yaw, pitch));
      gl!.uniform1f(U(surf, 'uReveal'), easeOut(reveal));
      gl!.drawElements(gl!.TRIANGLES, mesh.idx.length, gl!.UNSIGNED_INT, 0);

      gl!.enable(gl!.BLEND);
      gl!.blendFunc(gl!.ONE, gl!.ONE_MINUS_SRC_ALPHA);
      const show = alpha * Math.min(1, Math.max(0, (reveal - 0.6) / 0.4));
      balls.forEach((b, k) => {
        const tr = b.trail, n = tr.length / 2;
        for (let i = 0; i < n; i++) {
          const x = tr[2 * i], y = tr[2 * i + 1];
          scratch.set([x, H * loss(x, y) + 0.03, y, show * (0.1 + 0.9 * (i / n))], 4 * i);
        }
        drawLines(scratch, n, gl!.POINTS, col.balls[k], 3.2 * d); // breadcrumb trail
      });
      gl!.disable(gl!.DEPTH_TEST);
      balls.forEach((b, k) => {
        scratch.set([b.x, H * loss(b.x, b.y) + 0.07, b.y, show], 0);
        drawLines(scratch, 1, gl!.POINTS, col.balls[k], 13 * d);
      });
      gl!.disable(gl!.BLEND);

      // HTML overlays: optimizer tags and axis labels
      const placed: number[][] = [];
      balls.forEach((b, k) => {
        const s = project(vp, [b.x, H * loss(b.x, b.y) + 0.07, b.y]), el = tags[k];
        if (!el) return;
        el.style.opacity = s ? String(show) : '0';
        if (!s) return;
        const x = s[0] * canvas.clientWidth, y = s[1] * canvas.clientHeight;
        // stack tags of balls that sit on top of each other
        const stack = placed.filter((p) => Math.abs(p[0] - x) < 60 && Math.abs(p[1] - y) < 14).length;
        placed.push([x, y]);
        el.style.transform = `translate(${x}px, ${y - stack * 18}px)`;
      });
      AXES.forEach((p, k) => {
        const s = project(vp, p), el = axes[k];
        if (!el) return;
        el.style.opacity = s ? String(easeOut(reveal)) : '0';
        if (s) el.style.transform = `translate(${s[0] * canvas.clientWidth}px, ${s[1] * canvas.clientHeight}px)`;
      });
    }

    // ---- loop ------------------------------------------------------------------------------
    let raf = 0, last = 0, inView = false;
    function frameFn(now: number) {
      raf = 0;
      const dt = Math.min(0.05, last ? (now - last) / 1000 : 0);
      last = now;
      const motion = !reduced.matches;
      if (reveal < 1) reveal = Math.min(1, reveal + (dt * 1000) / REVEAL_MS);
      if (!dragging) {
        yawVel *= Math.pow(0.04, dt); pitchVel = 0;
        yaw += yawVel + (motion ? AUTO * dt : 0);
      }
      if (reveal > 0.6) {
        phaseT += dt * 1000;
        if (phase === 'run') {
          alpha = Math.min(1, alpha + dt * 3);
          stepAcc += dt * STEPS_PER_S;
          for (; stepAcc >= 1; stepAcc--) balls.forEach(stepBall);
          if (balls.every((b) => b.done)) { phase = 'hold'; phaseT = 0; if (userDrop) announce(); }
        } else if (phase === 'hold' && phaseT > HOLD_MS && motion) { phase = 'fade'; phaseT = 0; }
        else if (phase === 'fade') {
          alpha = Math.max(0, 1 - phaseT / FADE_MS);
          if (alpha === 0) { run++; userDrop = false; start(...startPoint(run)); }
        }
      }
      render();
      const busy = motion || reveal < 1 || dragging || Math.abs(yawVel) > 1e-4 || phase === 'run';
      if (busy && inView && !document.hidden) raf = requestAnimationFrame(frameFn);
      else last = 0; // idle: next wake starts with a fresh clock
    }
    function wake() { if (inView && !document.hidden && !raf) raf = requestAnimationFrame(frameFn); }
    const easeOut = (t: number) => 1 - (1 - t) ** 3;

    // ---- environment -----------------------------------------------------------------------
    const ro = new ResizeObserver(() => {
      const d = Math.min(2, devicePixelRatio || 1), r = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.round(r.width * d));
      canvas.height = Math.max(1, Math.round(r.height * d));
      aspect = r.width / Math.max(1, r.height);
      render();
    });
    ro.observe(canvas);
    const io = new IntersectionObserver(([e]) => { inView = e.isIntersecting; wake(); });
    io.observe(root);
    const onVis = () => wake();
    document.addEventListener('visibilitychange', onVis);
    const onTheme = () => { col = read(); render(); };
    const mo = new MutationObserver(onTheme);
    mo.observe(html, { attributes: true, attributeFilter: ['data-theme'] });
    const dark = matchMedia('(prefers-color-scheme: dark)');
    dark.addEventListener('change', onTheme);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect(); io.disconnect(); mo.disconnect();
      document.removeEventListener('visibilitychange', onVis);
      dark.removeEventListener('change', onTheme);
      gl.getExtension('WEBGL_lose_context')?.loseContext();
    };
  });
</script>

<figure class="land" bind:this={root}>
  <div
    class="stage"
    role="img"
    aria-label="A 3D loss landscape with a deep valley, a shallower valley and a ridge between them. Three optimizers, SGD, Momentum and Adam, roll downhill from the same starting point, leaving trails."
  >
    <canvas bind:this={canvas} aria-hidden="true"></canvas>
    {#each NAMES as name, i (name)}
      <span class="tag" bind:this={tags[i]} data-k={i}>{name}</span>
    {/each}
    <span class="axis" bind:this={axes[0]}>w₁</span>
    <span class="axis" bind:this={axes[1]}>w₂</span>
    {#if failed}
      <p class="fallback">This figure needs WebGL2, which your browser has turned off.</p>
    {/if}
    <p class="sr-only" aria-live="polite">{live}</p>
  </div>
  <figcaption>
    <span class="legend">
      {#each NAMES as name, i (name)}<span class="key" data-k={i}><i></i>{name}</span>{/each}
    </span>
    <span class="cap">
      A loss landscape. Each ball is an optimizer walking downhill from the same start. Drag to turn it; click the
      surface to drop new balls.
    </span>
    <button class="again" type="button" onclick={() => drop()}>New start</button>
  </figcaption>
</figure>

<style>
  .land { margin: 0; }
  .stage { position: relative; aspect-ratio: 4 / 3; touch-action: pan-y; }
  canvas { display: block; width: 100%; height: 100%; cursor: grab; }
  canvas:active { cursor: grabbing; }
  .tag, .axis {
    position: absolute; left: 0; top: 0; pointer-events: none; white-space: nowrap; opacity: 0;
    font: 500 0.6875rem / 1 var(--font-code); will-change: transform;
  }
  .tag { margin: -1.9em 0 0 0.7em; padding: 0.2em 0.4em; border-radius: var(--r-1); background: var(--bg); border: 1px solid currentColor; }
  .axis { margin: -0.5em 0 0 -0.5em; color: var(--text-muted); font-style: italic; }
  [data-k='0'] { color: var(--viz-neg); }
  [data-k='1'] { color: var(--viz-forward); }
  [data-k='2'] { color: var(--accent); }
  .fallback { position: absolute; inset: auto 0 45% 0; text-align: center; color: var(--text-muted); font: 0.875rem var(--font-ui); }
  figcaption { display: flex; flex-wrap: wrap; gap: var(--s-2) var(--s-4); align-items: baseline; margin-top: var(--s-2); font: 0.8125rem / 1.5 var(--font-ui); color: var(--text-muted); }
  .legend { display: flex; gap: var(--s-3); }
  .key { display: inline-flex; align-items: center; gap: 0.35rem; font-weight: 600; }
  .key i { width: 0.55rem; height: 0.55rem; border-radius: 50%; background: currentColor; }
  .cap { flex: 1 1 18rem; }
  .again { font: 600 0.75rem var(--font-ui); color: var(--text); background: none; border: 1px solid var(--rule); border-radius: var(--r-1); padding: 0.25rem 0.6rem; cursor: pointer; }
  .again:hover { border-color: var(--accent); }
  @media (min-width: 48em) { .stage { aspect-ratio: 6 / 5; } }
</style>
