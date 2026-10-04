"use client";

import { useEffect, useState } from "react";

import type { Quote } from "@/lib/quotes";

export interface Settings {
  sound: boolean;
  crt: boolean;
  typewriter: boolean;
}

export interface QuestState {
  xp: number;
  seen: string[];
  favorites: string[];
  custom: Quote[];
  settings: Settings;
}

const KEY = "quote-quest:v1";

export const INITIAL_STATE: QuestState = {
  xp: 0,
  seen: [],
  favorites: [],
  custom: [],
  settings: { sound: true, crt: true, typewriter: true },
};

export function loadState(): QuestState {
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return INITIAL_STATE;
    const parsed = JSON.parse(raw) as Partial<QuestState>;
    return {
      ...INITIAL_STATE,
      ...parsed,
      settings: { ...INITIAL_STATE.settings, ...parsed.settings },
    };
  } catch {
    return INITIAL_STATE;
  }
}

/**
 * Game progress, persisted to localStorage. Storage is only read once the
 * player presses start, so server and client render the same start screen.
 */
export function useQuestState() {
  const [state, setState] = useState<QuestState>(INITIAL_STATE);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    if (!loaded) return;
    try {
      window.localStorage.setItem(KEY, JSON.stringify(state));
    } catch {
      // Storage unavailable (private mode, quota): progress just won't persist.
    }
  }, [state, loaded]);

  const load = () => {
    const saved = loadState();
    setState(saved);
    setLoaded(true);
    return saved;
  };

  return { state, setState, load };
}

export const XP_PER_LEVEL = 100;

export function levelOf(xp: number) {
  return Math.floor(xp / XP_PER_LEVEL) + 1;
}
