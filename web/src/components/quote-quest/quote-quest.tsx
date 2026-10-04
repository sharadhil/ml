"use client";

import { useEffect, useEffectEvent, useMemo, useRef, useState } from "react";

import { Badge } from "@/components/ui/8bit/badge";
import { Button } from "@/components/ui/8bit/button";
import { Card, CardContent } from "@/components/ui/8bit/card";
import { Kbd } from "@/components/ui/8bit/kbd";
import ManaBar from "@/components/ui/8bit/mana-bar";
import { toast } from "@/components/ui/8bit/toast";
import {
  ToggleGroup,
  ToggleGroupItem,
} from "@/components/ui/8bit/toggle-group";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/8bit/tooltip";
import XpBar from "@/components/ui/8bit/xp-bar";
import { RetroModeSwitcher } from "@/components/ui/retro-mode-switcher";
import { useTypewriter } from "@/hooks/use-typewriter";
import {
  CATEGORIES,
  CATEGORY_BY_ID,
  type Category,
  formatQuote,
  type Quote,
  QUOTES,
} from "@/lib/quotes";
import { type SfxName, sfx } from "@/lib/sfx";
import { cn } from "@/lib/utils";

import { downloadQuoteCard } from "./export-card";
import { ForgeDialog } from "./forge-dialog";
import { InventorySheet } from "./inventory-sheet";
import { PixelSprite, type SpriteName } from "./pixel-sprite";
import { Starfield } from "./starfield";
import {
  INITIAL_STATE,
  levelOf,
  type QuestState,
  type Settings,
  useQuestState,
  XP_PER_LEVEL,
} from "./use-quest-state";

type Filter = Category | "all";

const XP_NEW = 10;
const XP_REPEAT = 3;
const XP_FORGE = 25;

function spriteFor(q: Quote): SpriteName {
  return q.custom ? "scroll" : q.category;
}

function isTyping(target: EventTarget | null) {
  return (
    target instanceof HTMLElement &&
    (target.isContentEditable ||
      ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName))
  );
}

export function QuoteQuest() {
  const { state, setState, load } = useQuestState();
  const { settings } = state;

  const [started, setStarted] = useState(false);
  const [filter, setFilter] = useState<Filter>("all");
  const [history, setHistory] = useState<string[]>([]);
  const [cursor, setCursor] = useState(-1);
  const [sheetOpen, setSheetOpen] = useState(false);
  const [forgeOpen, setForgeOpen] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [celebration, setCelebration] = useState(0);
  const [hit, setHit] = useState(0);

  const allQuotes = useMemo(() => [...QUOTES, ...state.custom], [state.custom]);
  const byId = useMemo(
    () => new Map(allQuotes.map((q) => [q.id, q])),
    [allQuotes],
  );
  const current = cursor >= 0 ? byId.get(history[cursor]) : undefined;
  const meta = current ? CATEGORY_BY_ID[current.category] : undefined;
  const accent = meta?.color ?? "#facc15";

  const level = levelOf(state.xp);
  const isFav = current ? state.favorites.includes(current.id) : false;

  const play = (name: SfxName, s: Settings = settings) => {
    if (s.sound) sfx[name]();
  };

  const tickRef = useRef(0);
  const { shown, done, skip } = useTypewriter(current?.text ?? "", {
    enabled: settings.typewriter && started,
    onChar: (char) => {
      if (char !== " " && tickRef.current++ % 2 === 0) play("blip");
    },
  });

  /** Moves the stage to `q`, awarding XP and announcing milestones. */
  const reveal = (
    q: Quote,
    base: QuestState = state,
    opts: { bonusXp?: number } = {},
  ) => {
    const fresh = !base.seen.includes(q.id);
    const gained = (fresh ? XP_NEW : XP_REPEAT) + (opts.bonusXp ?? 0);
    const before = levelOf(base.xp);
    const next: QuestState = {
      ...base,
      xp: base.xp + gained,
      seen: fresh ? [...base.seen, q.id] : base.seen,
    };
    setState(next);
    setHistory((h) => [...h.slice(0, cursor + 1), q.id].slice(-50));
    setCursor((c) => Math.min(c + 1, 49));
    setHit((n) => n + 1);
    stopSpeaking();

    const after = levelOf(next.xp);
    if (after > before) {
      play("levelUp", next.settings);
      setCelebration((n) => n + 1);
      toast(`LEVEL UP! You reached LV ${after}`);
    } else {
      play("coin", next.settings);
    }

    if (fresh && !q.custom) {
      const inCategory = QUOTES.filter((x) => x.category === q.category);
      const seenNow = new Set(next.seen);
      if (inCategory.every((x) => seenNow.has(x.id))) {
        toast(
          `${CATEGORY_BY_ID[q.category].label.toUpperCase()} CLEARED! All ${inCategory.length} found`,
        );
      }
      if (QUOTES.every((x) => seenNow.has(x.id))) {
        toast("100% COMPLETE! You found every quote");
      }
    }
  };

  const pickRandom = (base: QuestState, f: Filter = filter) => {
    const pool = [...QUOTES, ...base.custom].filter(
      (q) => (f === "all" || q.category === f) && q.id !== current?.id,
    );
    const unseen = pool.filter((q) => !base.seen.includes(q.id));
    const from = unseen.length ? unseen : pool;
    return from[Math.floor(Math.random() * from.length)];
  };

  const start = () => {
    sfx.unlock();
    const saved = load();
    setStarted(true);
    if (saved.settings.sound) sfx.start();
    const first = pickRandom(saved);
    if (first) window.setTimeout(() => reveal(first, saved), 450);
  };

  const next = () => {
    if (!started) return start();
    if (!done) return skip();
    if (cursor < history.length - 1) {
      setCursor(cursor + 1);
      setHit((n) => n + 1);
      play("select");
      return;
    }
    const q = pickRandom(state);
    if (q) reveal(q);
  };

  const prev = () => {
    if (cursor <= 0) return;
    setCursor(cursor - 1);
    setHit((n) => n + 1);
    play("back");
    stopSpeaking();
  };

  const chooseFilter = (value: string) => {
    if (!value) return;
    const f = value as Filter;
    setFilter(f);
    if (!started) return;
    play("select");
    const q = pickRandom(state, f);
    if (q && (f === "all" || current?.category !== f)) reveal(q);
  };

  const toggleFavorite = (id = current?.id) => {
    if (!id) return;
    const has = state.favorites.includes(id);
    setState({
      ...state,
      favorites: has
        ? state.favorites.filter((f) => f !== id)
        : [...state.favorites, id],
    });
    play(has ? "unsave" : "save");
    if (!has) toast("Saved to your inventory");
  };

  const copy = async () => {
    if (!current) return;
    try {
      await navigator.clipboard.writeText(formatQuote(current));
      play("copy");
      toast("Copied to clipboard");
    } catch {
      toast("Couldn't access the clipboard");
    }
  };

  const stopSpeaking = () => {
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
    setSpeaking(false);
  };

  const speak = () => {
    if (!current || !("speechSynthesis" in window)) {
      toast("Speech isn't supported in this browser");
      return;
    }
    if (speaking) return stopSpeaking();
    const u = new SpeechSynthesisUtterance(
      `${current.text} ... ${current.author}`,
    );
    u.rate = 0.95;
    u.pitch = 1.1;
    u.onend = () => setSpeaking(false);
    u.onerror = () => setSpeaking(false);
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(u);
    setSpeaking(true);
  };

  const share = () => {
    if (!current) return;
    const text = encodeURIComponent(`${formatQuote(current)}\n\n#QuoteQuest`);
    window.open(
      `https://x.com/intent/tweet?text=${text}`,
      "_blank",
      "noopener,noreferrer",
    );
  };

  const exportPng = async () => {
    if (!current) return;
    await downloadQuoteCard(current, spriteFor(current));
    play("copy");
    toast("Quote card downloaded");
  };

  const forge = (q: Omit<Quote, "id" | "custom">) => {
    const quote: Quote = { ...q, id: `custom-${Date.now()}`, custom: true };
    const base = { ...state, custom: [...state.custom, quote] };
    sfx.unlock();
    reveal(quote, base, { bonusXp: XP_FORGE });
    play("forge", base.settings);
    toast(`Quote forged! +${XP_FORGE} bonus XP`);
  };

  const updateSettings = (patch: Partial<Settings>) =>
    setState({ ...state, settings: { ...settings, ...patch } });

  const resetProgress = () => {
    setState({ ...INITIAL_STATE, settings });
    setHistory([]);
    setCursor(-1);
    setStarted(false);
    setSheetOpen(false);
    stopSpeaking();
    toast("Progress reset. Press start!");
  };

  const show = (id: string) => {
    const q = byId.get(id);
    if (!q) return;
    setSheetOpen(false);
    reveal(q);
  };

  const onKey = useEffectEvent((e: KeyboardEvent) => {
    if (
      e.metaKey ||
      e.ctrlKey ||
      e.altKey ||
      isTyping(e.target) ||
      sheetOpen ||
      forgeOpen
    ) {
      return;
    }
    const key = e.key.toLowerCase();
    const actions: Record<string, () => void> = {
      " ": next,
      enter: next,
      arrowright: next,
      n: next,
      arrowleft: prev,
      b: prev,
      s: () => toggleFavorite(),
      c: () => void copy(),
      r: speak,
      p: () => void exportPng(),
      f: () => setForgeOpen(true),
      i: () => setSheetOpen(true),
      m: () => {
        updateSettings({ sound: !settings.sound });
        toast(settings.sound ? "Sound off" : "Sound on");
      },
    };
    const digit = Number.parseInt(key, 10);
    if (key.length === 1 && digit >= 0 && digit <= CATEGORIES.length) {
      e.preventDefault();
      chooseFilter(digit === 0 ? "all" : CATEGORIES[digit - 1].id);
      return;
    }
    const action = actions[key];
    if (!action) return;
    if (!started && !["enter", " "].includes(key)) return;
    e.preventDefault();
    action();
  });

  useEffect(() => {
    const handler = (e: KeyboardEvent) => onKey(e);
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  useEffect(() => () => window.speechSynthesis?.cancel(), []);

  return (
    <main className="retro relative mx-auto flex w-full max-w-4xl flex-1 flex-col gap-6 px-4 py-6 sm:py-10">
      {/* Title bar */}
      <header className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <PixelSprite
            name="scroll"
            accent="#facc15"
            className="size-10 animate-bob"
          />
          <h1 className="title-shadow text-lg leading-none sm:text-2xl">
            QUOTE <span style={{ color: accent }}>QUEST</span>
          </h1>
        </div>
        <div className="flex items-center gap-3">
          {started && (
            <>
              <div>
                <Tooltip>
                  <TooltipTrigger>
                    <Button
                      variant="ghost"
                      size="icon"
                      aria-label={
                        settings.sound ? "Mute sound" : "Unmute sound"
                      }
                      onClick={() => updateSettings({ sound: !settings.sound })}
                    >
                      <SoundIcon on={settings.sound} />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Sound (M)</TooltipContent>
                </Tooltip>
              </div>
              <InventorySheet
                open={sheetOpen}
                onOpenChange={setSheetOpen}
                favorites={state.favorites
                  .map((id) => byId.get(id))
                  .filter((q): q is Quote => !!q)}
                log={[...history]
                  .reverse()
                  .map((id) => byId.get(id))
                  .filter((q): q is Quote => !!q)}
                settings={settings}
                stats={{
                  xp: state.xp,
                  level,
                  seen: state.seen.length,
                  total: allQuotes.length,
                  forged: state.custom.length,
                }}
                onShow={show}
                onRemove={toggleFavorite}
                onSettings={updateSettings}
                onReset={resetProgress}
              />
            </>
          )}
          <RetroModeSwitcher />
        </div>
      </header>

      {/* Category select */}
      <ToggleGroup
        type="single"
        value={filter}
        onValueChange={chooseFilter}
        className="flex flex-wrap justify-center gap-3 sm:gap-4"
        aria-label="Quote category"
      >
        <ToggleGroupItem
          value="all"
          variant="outline"
          className="relative h-9 px-3 text-[10px] sm:text-xs"
        >
          ALL
        </ToggleGroupItem>
        {CATEGORIES.map((c) => (
          <ToggleGroupItem
            key={c.id}
            value={c.id}
            variant="outline"
            className="relative h-9 gap-2 px-3 text-[10px] sm:text-xs"
            style={
              filter === c.id
                ? { background: c.color, color: "#0b0b14" }
                : undefined
            }
          >
            <PixelSprite name={c.id} accent={c.color} className="size-5" />
            {c.label.toUpperCase()}
          </ToggleGroupItem>
        ))}
      </ToggleGroup>

      {/* Cabinet */}
      <Card className="relative">
        <CardContent className="flex flex-col gap-5 p-4 sm:p-6">
          {/* HUD */}
          <div className="grid gap-4 text-[10px] sm:grid-cols-2 sm:text-xs">
            <div className="flex flex-col gap-2">
              <div className="flex items-center justify-between">
                <span>LV {started ? level : "-"}</span>
                <span className="text-muted-foreground">
                  {started
                    ? `${state.xp % XP_PER_LEVEL}/${XP_PER_LEVEL} XP`
                    : "INSERT COIN"}
                </span>
              </div>
              <XpBar value={started ? state.xp % XP_PER_LEVEL : 0} />
            </div>
            <div className="flex flex-col gap-2">
              <div className="flex items-center justify-between">
                <span>COLLECTION</span>
                <span className="text-muted-foreground">
                  {started ? `${state.seen.length}/${allQuotes.length}` : "-"}
                </span>
              </div>
              <ManaBar
                value={
                  started
                    ? Math.round((state.seen.length / allQuotes.length) * 100)
                    : 0
                }
              />
            </div>
          </div>

          {/* Screen */}
          <div
            className={cn(
              "screen relative isolate overflow-hidden",
              settings.crt && "crt",
            )}
            style={{ "--qq-accent": accent } as React.CSSProperties}
          >
            <Starfield />
            {celebration > 0 && <Confetti key={celebration} />}

            {!started || !current ? (
              <StartScreen onStart={start} starting={started} />
            ) : (
              <div
                key={hit}
                className="relative z-10 flex min-h-[340px] flex-col justify-end gap-4 p-4 sm:min-h-[360px] sm:p-6"
              >
                <div className="flex items-center justify-between gap-2">
                  <Badge
                    className="text-[10px]"
                    style={{ background: accent, color: "#0b0b14" }}
                  >
                    {current.custom ? "FORGED" : meta?.label.toUpperCase()}
                  </Badge>
                  <span className="text-[10px] text-white/60">
                    #{String(allQuotes.indexOf(current) + 1).padStart(3, "0")}
                  </span>
                </div>

                <div className="flex flex-col items-center gap-4 sm:flex-row sm:items-end">
                  <div className="flex shrink-0 flex-col items-center gap-2">
                    <span className="text-[9px] text-[var(--qq-accent)]">
                      {current.custom ? "SCRIBE" : meta?.hero.toUpperCase()}
                    </span>
                    <PixelSprite
                      name={spriteFor(current)}
                      accent={accent}
                      label={current.custom ? "Scroll" : meta?.hero}
                      className={cn(
                        "size-24 animate-hop sm:size-28",
                        !done && "animate-talk",
                      )}
                    />
                    <div className="ground h-2 w-24 sm:w-28" />
                  </div>

                  <button
                    type="button"
                    onClick={next}
                    className="dialogue group relative w-full cursor-pointer p-4 text-left text-white sm:p-5"
                    aria-label={done ? "Next quote" : "Skip typing"}
                  >
                    <span className="sr-only" aria-live="polite">
                      {done ? formatQuote(current) : ""}
                    </span>
                    <span
                      aria-hidden
                      className="grid text-xs leading-[1.9] sm:text-sm"
                    >
                      <span className="invisible col-start-1 row-start-1">
                        &ldquo;{current.text}&rdquo;
                      </span>
                      <span className="col-start-1 row-start-1">
                        &ldquo;{shown}
                        {done ? <>&rdquo;</> : <span className="caret" />}
                      </span>
                    </span>
                    <span
                      aria-hidden
                      className={cn(
                        "mt-3 flex items-end justify-between gap-2 text-[10px] transition-opacity sm:text-xs",
                        done ? "opacity-100" : "opacity-0",
                      )}
                    >
                      <span className="text-[var(--qq-accent)]">
                        - {current.author}
                        {current.source && (
                          <span className="text-white/60">
                            , {current.source}
                          </span>
                        )}
                      </span>
                      <span className="animate-blink text-white">&#9660;</span>
                    </span>
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Controls */}
          <div className="flex flex-col gap-5">
            <div className="flex items-center gap-4">
              <Button
                variant="outline"
                onClick={prev}
                disabled={!started || cursor <= 0}
                aria-label="Previous quote"
              >
                &#9664;
              </Button>
              <Button
                className="h-12 flex-1 text-xs sm:text-sm"
                onClick={next}
                style={{ background: accent, color: "#0b0b14" }}
              >
                {!started ? "PRESS START" : !done ? "SKIP >>" : "NEXT QUOTE"}{" "}
                &#9654;
              </Button>
            </div>
            <div className="grid grid-cols-3 gap-4 sm:grid-cols-6">
              <ActionButton
                label="SAVE"
                hint="S"
                onClick={() => toggleFavorite()}
                disabled={!current}
                active={isFav}
              >
                <HeartIcon filled={isFav} />
              </ActionButton>
              <ActionButton
                label="COPY"
                hint="C"
                onClick={() => void copy()}
                disabled={!current}
              >
                <CopyIcon />
              </ActionButton>
              <ActionButton
                label={speaking ? "STOP" : "READ"}
                hint="R"
                onClick={speak}
                disabled={!current}
                active={speaking}
              >
                <SoundIcon on />
              </ActionButton>
              <ActionButton
                label="POST"
                hint="Share on X"
                onClick={share}
                disabled={!current}
              >
                <ShareIcon />
              </ActionButton>
              <ActionButton
                label="PNG"
                hint="P"
                onClick={() => void exportPng()}
                disabled={!current}
              >
                <ImageIcon />
              </ActionButton>
              <ForgeDialog
                open={forgeOpen}
                onOpenChange={setForgeOpen}
                onForge={forge}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Controls legend */}
      <footer className="flex flex-wrap items-center justify-center gap-x-5 gap-y-3 text-[9px] text-muted-foreground sm:text-[10px]">
        <Legend keys={["SPACE"]} label="next / skip" />
        <Legend keys={["◀", "▶"]} label="browse" />
        <Legend keys={["0", "5"]} sep="-" label="category" />
        <Legend keys={["S"]} label="save" />
        <Legend keys={["C"]} label="copy" />
        <Legend keys={["F"]} label="forge" />
        <Legend keys={["I"]} label="inventory" />
        <Legend keys={["D"]} label="day/night" />
      </footer>
    </main>
  );
}

function StartScreen({
  onStart,
  starting,
}: {
  onStart: () => void;
  starting: boolean;
}) {
  return (
    <div className="relative z-10 flex min-h-[340px] flex-col items-center justify-center gap-6 p-6 text-center text-white sm:min-h-[360px]">
      <div className="flex items-end gap-3 sm:gap-5">
        {CATEGORIES.map((c, i) => (
          <PixelSprite
            key={c.id}
            name={c.id}
            accent={c.color}
            className="size-10 animate-hop sm:size-14"
            label={c.hero}
            style={{ animationDelay: `${i * 120}ms` }}
          />
        ))}
      </div>
      <h2 className="title-shadow text-2xl leading-tight sm:text-4xl">
        QUOTE
        <br />
        QUEST
      </h2>
      <p className="max-w-sm text-[10px] leading-relaxed text-white/70 sm:text-xs">
        Collect all {QUOTES.length} legendary quotes. Level up. Forge your own.
      </p>
      <button
        type="button"
        onClick={onStart}
        disabled={starting}
        className="animate-blink cursor-pointer text-xs text-yellow-300 sm:text-sm"
      >
        {starting ? "LOADING..." : "> PRESS START <"}
      </button>
      <p className="text-[9px] text-white/40">&copy; 1986 NOT A REAL COMPANY</p>
    </div>
  );
}

function ActionButton({
  label,
  hint,
  onClick,
  disabled,
  active,
  children,
}: {
  label: string;
  hint: string;
  onClick: () => void;
  disabled?: boolean;
  active?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div>
      <Tooltip>
        <TooltipTrigger>
          <Button
            variant={active ? "default" : "outline"}
            onClick={onClick}
            disabled={disabled}
            className="h-11 w-full gap-2 text-[10px]"
            aria-pressed={active}
          >
            {children}
            {label}
          </Button>
        </TooltipTrigger>
        <TooltipContent>
          {hint.length === 1 ? `Shortcut: ${hint}` : hint}
        </TooltipContent>
      </Tooltip>
    </div>
  );
}

function Legend({
  keys,
  label,
  sep = "",
}: {
  keys: string[];
  label: string;
  sep?: string;
}) {
  return (
    <span className="flex items-center gap-1.5">
      {keys.map((k, i) => (
        <span key={k} className="flex items-center gap-1.5">
          {i > 0 && sep}
          <Kbd className="text-[9px]">{k}</Kbd>
        </span>
      ))}
      <span>{label}</span>
    </span>
  );
}

function Confetti() {
  const [bits] = useState(() =>
    Array.from({ length: 36 }, (_, i) => ({
      left: Math.random() * 100,
      delay: Math.random() * 0.3,
      dur: 0.9 + Math.random() * 0.8,
      color: CATEGORIES[i % CATEGORIES.length].color,
      drift: (Math.random() - 0.5) * 120,
    })),
  );
  return (
    <div
      className="pointer-events-none absolute inset-0 z-20 overflow-hidden"
      aria-hidden
    >
      <div className="level-flash absolute inset-0" />
      {bits.map((b, i) => (
        <span
          key={i}
          className="confetti absolute top-0 size-2"
          style={
            {
              left: `${b.left}%`,
              background: b.color,
              animationDelay: `${b.delay}s`,
              animationDuration: `${b.dur}s`,
              "--drift": `${b.drift}px`,
            } as React.CSSProperties
          }
        />
      ))}
    </div>
  );
}

/* Pixel icons drawn on an 8x8 grid. */
function PixelIcon({
  rows,
  className,
}: {
  rows: string[];
  className?: string;
}) {
  return (
    <svg
      viewBox="0 0 8 8"
      shapeRendering="crispEdges"
      className={cn("size-4 shrink-0", className)}
      aria-hidden
    >
      {rows.flatMap((row, y) =>
        [...row].map((c, x) =>
          c === "#" ? (
            <rect
              key={`${x}-${y}`}
              x={x}
              y={y}
              width={1}
              height={1}
              fill="currentColor"
            />
          ) : null,
        ),
      )}
    </svg>
  );
}

function HeartIcon({ filled }: { filled: boolean }) {
  return (
    <PixelIcon
      className={filled ? "text-red-500" : undefined}
      rows={
        filled
          ? [
              ".##.##..",
              "#######.",
              "#######.",
              "#######.",
              ".#####..",
              "..###...",
              "...#....",
              "........",
            ]
          : [
              ".##.##..",
              "#..#..#.",
              "#.....#.",
              "#.....#.",
              ".#...#..",
              "..#.#...",
              "...#....",
              "........",
            ]
      }
    />
  );
}

function CopyIcon() {
  return (
    <PixelIcon
      rows={[
        "#####...",
        "#...#...",
        "#.#####.",
        "#.#...#.",
        "###...#.",
        "..#...#.",
        "..#####.",
        "........",
      ]}
    />
  );
}

function SoundIcon({ on }: { on: boolean }) {
  return (
    <PixelIcon
      rows={
        on
          ? [
              "...#....",
              "..##..#.",
              "####.#..",
              "####.#.#",
              "####.#.#",
              "####.#..",
              "..##..#.",
              "...#....",
            ]
          : [
              "...#....",
              "..##....",
              "####.#.#",
              "####..#.",
              "####..#.",
              "####.#.#",
              "..##....",
              "...#....",
            ]
      }
    />
  );
}

function ShareIcon() {
  return (
    <PixelIcon
      rows={[
        "....##..",
        "....###.",
        "#######.",
        "#...###.",
        "#...##..",
        "#.......",
        "#.......",
        "######..",
      ]}
    />
  );
}

function ImageIcon() {
  return (
    <PixelIcon
      rows={[
        "########",
        "#......#",
        "#.##...#",
        "#.##...#",
        "#....#.#",
        "#.#.###.",
        "#######.",
        "........",
      ]}
    />
  );
}
