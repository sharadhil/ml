import react from '@vitejs/plugin-react';
import { wgslVitePlugin } from 'vgpu/client';
import { defineConfig } from 'vite';

export default defineConfig({
  // wgslVitePlugin turns `import shader from './x.wgsl'` into a typed vgpu
  // ShaderSource, which is what the black-hole pipeline's `effect()` calls expect.
  plugins: [wgslVitePlugin(), react()],
});
