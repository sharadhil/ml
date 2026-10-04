import { useEffect, useRef } from 'react';
import { createRenderer } from './renderer';

export type BlackHoleStatus = 'loading' | 'ready' | 'error';

interface ExampleProps {
  className?: string;
  /** Reports when the first GPU frame is ready, or when WebGPU is unavailable. */
  onStatusChange?: (status: BlackHoleStatus, error?: unknown) => void;
}

export function Example({ className, onStatusChange }: ExampleProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const onStatusRef = useRef(onStatusChange);
  onStatusRef.current = onStatusChange;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let active = true;
    const renderer = createRenderer({ canvas });
    onStatusRef.current?.('loading');
    renderer.ready.then(
      () => {
        if (active) onStatusRef.current?.('ready');
      },
      (error: unknown) => {
        if (active) onStatusRef.current?.('error', error);
      },
    );

    return () => {
      active = false;
      renderer.dispose();
    };
  }, []);

  return (
    <div className={['black-hole', className].filter(Boolean).join(' ')}>
      <canvas ref={canvasRef} className="black-hole__canvas" />
    </div>
  );
}

export default Example;
