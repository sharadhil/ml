import {
  clock,
  frameLoop,
  surface,
  type Gpu,
  type Surface,
} from 'vgpu';

import {
  createEffects,
  createTargets,
  destroyTargets,
  prewarm,
  renderChain,
  setBindings,
  type Orbit,
} from './pipeline';

interface RendererOptions {
  canvas: HTMLCanvasElement;
}

interface RenderSize {
  width: number;
  height: number;
  dpr: number;
}

// The raymarch pass is the whole cost of a frame, so it renders into an
// internal target that tracks a fraction of the canvas resolution; the
// composite pass upscales it with linear filtering. The fraction adapts to
// measured frame times so every GPU settles near a smooth frame rate.
const START_PIXELS = 640_000; // first frames: ~1067x600, safe on integrated GPUs
const MIN_SCALE = 0.35;
const MAX_SCALE = 1;
const SLOW_FRAME_MS = 21; // below ~48 fps: step resolution down
const SMOOTH_FRAME_MS = 17.8; // keeping up with a 60 Hz display (or better)
const SMOOTH_WINDOWS_TO_GROW = 3; // ~1.5 s of smooth frames before stepping up
const SAMPLE_FRAMES = 30;
const SAMPLE_WINDOW_MS = 500; // or decide sooner when frames are this slow
const WARMUP_MS = 1000; // first frames after start (or a hidden tab) run slow while drivers settle

function sceneSize(size: RenderSize, scale: number): [number, number] {
  return [
    Math.max(1, Math.round(size.width * size.dpr * scale)),
    Math.max(1, Math.round(size.height * size.dpr * scale)),
  ];
}

function startScale(pixels: number): number {
  return Math.min(MAX_SCALE, Math.max(MIN_SCALE, Math.sqrt(START_PIXELS / Math.max(1, pixels))));
}

export function createRenderer(options: RendererOptions) {
  let disposed = false;
  let gpu: Gpu | undefined;
  let canvasSurface: Surface | undefined;
  let effects: ReturnType<typeof createEffects> | undefined;
  let targets: ReturnType<typeof createTargets> | undefined;
  let input: ReturnType<typeof installOrbitInput> | undefined;
  let observer: ResizeObserver | undefined;
  let resizeFrame = 0;
  let pendingSize: RenderSize | undefined;
  let lastDpr = typeof window === 'undefined' ? 1 : window.devicePixelRatio;
  let lastSize: RenderSize | undefined;
  let renderScale = MAX_SCALE;
  let scaleCeiling = MAX_SCALE; // lowered whenever a scale proves too slow
  let smoothWindows = 0;

  const applyResize = () => {
    resizeFrame = 0;
    const size = pendingSize;
    pendingSize = undefined;
    if (disposed || !size || !gpu || !effects || !targets || !canvasSurface) return;

    try {
      const previousTargets = targets;
      const nextTargets = createTargets(gpu, sceneSize(size, renderScale));

      try {
        setBindings(effects, nextTargets);
      } catch (error) {
        destroyTargets(nextTargets);
        throw error;
      }

      targets = nextTargets;
      destroyTargets(previousTargets);
    } catch (error) {
      fail(error);
    }
  };

  const resize = (size: RenderSize) => {
    if (disposed || size.width <= 0 || size.height <= 0) return;
    lastSize = size;
    pendingSize = size;
    if (!resizeFrame) resizeFrame = requestAnimationFrame(applyResize);
  };

  const measure = () => {
    const rect = options.canvas.getBoundingClientRect();
    resize({
      width: rect.width,
      height: rect.height,
      dpr: Math.min(1.6, Math.max(1, window.devicePixelRatio || 1)),
    });
  };

  const onWindowResize = () => {
    if (window.devicePixelRatio === lastDpr) return;
    lastDpr = window.devicePixelRatio;
    measure();
  };

  // Frame timing restarts after the tab was hidden, so that gap isn't read as GPU load.
  let resetTiming = () => {};
  const onVisibilityChange = () => resetTiming();

  const dispose = () => {
    if (disposed) return;
    disposed = true;
    if (resizeFrame) cancelAnimationFrame(resizeFrame);
    observer?.disconnect();
    if (typeof window !== 'undefined') {
      window.removeEventListener('resize', onWindowResize);
      document.removeEventListener('visibilitychange', onVisibilityChange);
    }
    input?.dispose();
    gpu?.dispose();
  };

  const initialize = async () => {
    const { init } = await import('vgpu');
    if (disposed) return;

    const nextGpu = await init();
    if (disposed) {
      nextGpu.dispose();
      return;
    }

    gpu = nextGpu;
    canvasSurface = surface(gpu, options.canvas, { dpr: [1, 1.6] });
    renderScale = startScale(canvasSurface.size[0] * canvasSurface.size[1]);
    targets = createTargets(gpu, sceneSize({ width: canvasSurface.size[0], height: canvasSurface.size[1], dpr: 1 }, renderScale));
    effects = createEffects(gpu, targets);
    setBindings(effects, targets);
    await prewarm(effects, targets, canvasSurface);
    if (disposed) return;

    input = installOrbitInput(options.canvas);
    observer = typeof ResizeObserver === 'undefined' ? undefined : new ResizeObserver(measure);
    observer?.observe(options.canvas);
    window.addEventListener('resize', onWindowResize);
    document.addEventListener('visibilitychange', onVisibilityChange);
    measure();

    const gpuClock = clock(gpu);
    let sampleMs = 0;
    let sampleCount = 0;
    let lastTick = performance.now();
    let sampleFrom = lastTick + WARMUP_MS;
    resetTiming = () => {
      lastTick = performance.now();
      sampleFrom = lastTick + WARMUP_MS;
      sampleMs = 0;
      sampleCount = 0;
    };
    frameLoop(gpu, (currentFrame) => {
      if (disposed || !effects || !targets || !canvasSurface || !input) return;

      const now = performance.now();
      const frameMs = now - lastTick;
      lastTick = now;
      // Frames that swap render targets pay a one-off allocation; leave them out.
      if (!resizeFrame && now >= sampleFrom) {
        sampleMs += frameMs;
        sampleCount += 1;
        if (sampleCount >= SAMPLE_FRAMES || sampleMs >= SAMPLE_WINDOW_MS) {
          adaptScale(sampleMs / sampleCount);
          sampleMs = 0;
          sampleCount = 0;
        }
      }

      effects.scene.set({
        params: { pointer: input.update(gpuClock.deltaTime), time: gpuClock.time },
      });
      renderChain(currentFrame, effects, targets, canvasSurface);
    });
  };

  function adaptScale(averageMs: number) {
    let next = renderScale;
    if (averageMs > SLOW_FRAME_MS) {
      scaleCeiling = Math.max(MIN_SCALE, renderScale * 0.95);
      next = Math.max(MIN_SCALE, renderScale * 0.8);
      smoothWindows = 0;
    } else if (averageMs < SMOOTH_FRAME_MS) {
      if (++smoothWindows >= SMOOTH_WINDOWS_TO_GROW) {
        next = Math.min(scaleCeiling, renderScale * 1.12);
        smoothWindows = 0;
      }
    } else {
      smoothWindows = 0;
    }
    if (Math.abs(next - renderScale) < 0.01 || !lastSize) return;
    renderScale = next;
    resize(lastSize);
  }

  function fail(error: unknown): never {
    dispose();
    throw error;
  }

  const ready = initialize().catch((error: unknown) => {
    if (disposed) return;
    fail(error);
  });

  return { ready, resize, dispose };
}

function installOrbitInput(canvas: HTMLCanvasElement) {
  let yaw = 0;
  let pitch = 0.05;
  let targetYaw = 0;
  let targetPitch = 0.05;
  let activePointer: number | undefined;
  const previousTouchAction = canvas.style.touchAction;
  canvas.style.touchAction = 'none';

  const down = (event: PointerEvent) => {
    if (!event.isPrimary || activePointer !== undefined) return;
    activePointer = event.pointerId;
    canvas.setPointerCapture?.(event.pointerId);
  };

  const move = (event: PointerEvent) => {
    if (!event.isPrimary || (activePointer !== undefined && event.pointerId !== activePointer)) {
      return;
    }

    const rect = canvas.getBoundingClientRect();
    const x = Math.max(
      0,
      Math.min(1, (event.clientX - rect.left) / Math.max(1, rect.width)),
    );
    const y = Math.max(
      0,
      Math.min(1, (event.clientY - rect.top) / Math.max(1, rect.height)),
    );
    targetYaw = (0.5 - x) * Math.PI * 1.4;
    targetPitch = Math.max(
      -Math.PI * 0.42,
      Math.min(Math.PI * 0.42, (y - 0.5) * Math.PI * 0.7),
    );
  };

  const end = (event: PointerEvent) => {
    if (event.pointerId !== activePointer) return;
    if (canvas.hasPointerCapture?.(event.pointerId)) {
      canvas.releasePointerCapture(event.pointerId);
    }
    activePointer = undefined;
  };

  canvas.addEventListener('pointerdown', down);
  canvas.addEventListener('pointermove', move);
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);

  return {
    update(deltaTime: number): Orbit {
      // Time-based easing (equal to the original 0.12/frame at 60 fps), so the
      // camera glides at the same speed whatever the frame rate.
      const ease = 1 - Math.exp(-Math.min(Math.max(deltaTime, 0), 0.1) * 7.67);
      yaw += (targetYaw - yaw) * ease;
      pitch += (targetPitch - pitch) * ease;
      return [yaw, pitch];
    },
    dispose() {
      canvas.removeEventListener('pointerdown', down);
      canvas.removeEventListener('pointermove', move);
      canvas.removeEventListener('pointerup', end);
      canvas.removeEventListener('pointercancel', end);
      if (activePointer !== undefined && canvas.hasPointerCapture?.(activePointer)) {
        canvas.releasePointerCapture(activePointer);
      }
      activePointer = undefined;
      canvas.style.touchAction = previousTouchAction;
    },
  };
}
