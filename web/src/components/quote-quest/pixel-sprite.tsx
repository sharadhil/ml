import { cn } from "@/lib/utils";

import { SPRITES } from "./sprites";

export type SpriteName = keyof typeof SPRITES;

export function shade(hex: string, amount: number) {
  const n = Number.parseInt(hex.slice(1), 16);
  const ch = (shift: number) =>
    Math.round(((n >> shift) & 255) * (1 - amount))
      .toString(16)
      .padStart(2, "0");
  return `#${ch(16)}${ch(8)}${ch(0)}`;
}

export function spritePalette(accent: string): Record<string, string> {
  return {
    k: "#0b0b14",
    a: accent,
    d: shade(accent, 0.45),
    w: "#f8fafc",
    s: "#f5c9a0",
    y: "#facc15",
    g: "#9ca3af",
    e: "#ffffff",
  };
}

/** Merges horizontal runs of the same colour into single rects. */
export function spriteRuns(name: SpriteName) {
  const runs: { x: number; y: number; w: number; key: string }[] = [];
  SPRITES[name].forEach((row, y) => {
    let x = 0;
    while (x < row.length) {
      const key = row[x];
      let w = 1;
      while (row[x + w] === key) w++;
      if (key !== ".") runs.push({ x, y, w, key });
      x += w;
    }
  });
  return runs;
}

interface PixelSpriteProps {
  name: SpriteName;
  accent: string;
  className?: string;
  label?: string;
  style?: React.CSSProperties;
}

export function PixelSprite({
  name,
  accent,
  className,
  label,
  style,
}: PixelSpriteProps) {
  const palette = spritePalette(accent);
  return (
    <svg
      viewBox="0 0 16 16"
      shapeRendering="crispEdges"
      className={cn("pixelated", className)}
      style={style}
      role={label ? "img" : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
    >
      {spriteRuns(name).map(({ x, y, w, key }) => (
        <rect
          key={`${x}-${y}`}
          x={x}
          y={y}
          width={w}
          height={1}
          fill={palette[key]}
        />
      ))}
    </svg>
  );
}
