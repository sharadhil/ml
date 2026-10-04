# UMBRA-1 — mission hero

A cinematic landing-page hero for a fictional NASA mission concept, built around
the vgpu **Black Hole** example (raymarched gravitational lensing, Doppler-beamed
accretion disk, HDR bloom chain).

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # typecheck + production build
```

Needs a WebGPU browser (Chrome/Edge 113+, Safari 26+). Elsewhere the hero
shows a CSS still of the black hole and says so.

## Layout

- `src/black-hole/` — the example, pulled with
  `npx vgpu examples pull black-hole` (revision `6fa27bb4…`).
  `renderer.ts`, `pipeline.ts` and all `.wgsl` shaders are unmodified.
  `index.tsx` is adapted: plain CSS classes instead of Tailwind, and an
  `onStatusChange` callback that reports `renderer.ready` instead of discarding it.
- `src/Hero.tsx` — copy, HUD, telemetry, letterbox reveal.
- `src/styles.css` — all styling, including the example's container classes.
- `vite.config.ts` — registers vgpu's `wgslVitePlugin` so `.wgsl` imports become
  typed `ShaderSource` modules.
