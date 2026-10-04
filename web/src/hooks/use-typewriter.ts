"use client";

import { useEffect, useEffectEvent, useState } from "react";

interface Options {
  enabled: boolean;
  /** Milliseconds per character. */
  speed?: number;
  onChar?: (char: string, index: number) => void;
}

/** RPG-style text reveal. Changing `text` restarts the animation. */
export function useTypewriter(
  text: string,
  { enabled, speed = 24, onChar }: Options,
) {
  const [state, setState] = useState({ text, count: 0 });
  const count = !enabled ? text.length : state.text === text ? state.count : 0;
  const done = count >= text.length;

  const emit = useEffectEvent((char: string, index: number) =>
    onChar?.(char, index),
  );

  useEffect(() => {
    if (!enabled || !text) return;
    const id = window.setInterval(() => {
      setState((prev) => {
        const current = prev.text === text ? prev.count : 0;
        if (current >= text.length) {
          window.clearInterval(id);
          return prev;
        }
        return { text, count: current + 1 };
      });
    }, speed);
    return () => window.clearInterval(id);
  }, [text, enabled, speed]);

  useEffect(() => {
    if (count > 0 && count <= text.length && enabled)
      emit(text[count - 1], count - 1);
  }, [count, text, enabled]);

  const skip = () => setState({ text, count: text.length });

  return { shown: text.slice(0, count), done, skip };
}
