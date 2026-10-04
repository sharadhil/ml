import react from '@vitejs/plugin-react';
import { wgslVitePlugin } from 'vgpu/client';
import { defineConfig } from 'vite';

export default defineConfig({
  // wgslVitePlugin turns `import shader from './x.wgsl'` into a typed vgpu
  // ShaderSource, which is what the black-hole pipeline's `effect()` calls expect.
  plugins: [wgslVitePlugin(), react()],
  // vgpu is also loaded via a dynamic import; pre-bundle it at dev-server start
  // so the first visit doesn't stall on dependency discovery and a reload.
  optimizeDeps: { include: ['vgpu'] },
});
