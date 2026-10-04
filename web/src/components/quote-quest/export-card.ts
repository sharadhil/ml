import { CATEGORY_BY_ID, type Quote } from "@/lib/quotes";

import { type SpriteName, spritePalette, spriteRuns } from "./pixel-sprite";

const W = 1200;
const H = 630;

function wrap(ctx: CanvasRenderingContext2D, text: string, maxWidth: number) {
  const lines: string[] = [];
  let line = "";
  for (const word of text.split(/\s+/)) {
    const next = line ? `${line} ${word}` : word;
    if (ctx.measureText(next).width > maxWidth && line) {
      lines.push(line);
      line = word;
    } else {
      line = next;
    }
  }
  if (line) lines.push(line);
  return lines;
}

/** Renders the quote as a 1200x630 pixel-art card and downloads it as PNG. */
export async function downloadQuoteCard(quote: Quote, sprite: SpriteName) {
  const accent = CATEGORY_BY_ID[quote.category].color;
  const family =
    getComputedStyle(document.documentElement)
      .getPropertyValue("--font-pixel")
      .trim() || "monospace";
  await document.fonts.load(`24px ${family}`).catch(() => undefined);

  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.imageSmoothingEnabled = false;

  // Night sky with a deterministic starfield.
  ctx.fillStyle = "#0b0b1e";
  ctx.fillRect(0, 0, W, H);
  let seed = quote.text.length * 9301 + 49297;
  const rand = () => (seed = (seed * 9301 + 49297) % 233280) / 233280;
  for (let i = 0; i < 120; i++) {
    ctx.fillStyle = i % 7 === 0 ? accent : "rgba(255,255,255,0.7)";
    const s = i % 11 === 0 ? 6 : 3;
    ctx.fillRect(Math.floor(rand() * W), Math.floor(rand() * H), s, s);
  }

  // Chunky pixel frame.
  ctx.fillStyle = accent;
  ctx.fillRect(24, 24, W - 48, 12);
  ctx.fillRect(24, H - 36, W - 48, 12);
  ctx.fillRect(24, 24, 12, H - 48);
  ctx.fillRect(W - 36, 24, 12, H - 48);

  // Hero sprite.
  const palette = spritePalette(accent);
  const px = 14;
  const sx = 80;
  const sy = H / 2 - 8 * px;
  for (const { x, y, w, key } of spriteRuns(sprite)) {
    ctx.fillStyle = palette[key];
    ctx.fillRect(sx + x * px, sy + y * px, w * px, px);
  }

  // Quote text, shrinking until it fits.
  const left = 360;
  const maxWidth = W - left - 80;
  let size = 34;
  let lines: string[] = [];
  do {
    ctx.font = `${size}px ${family}`;
    lines = wrap(ctx, `"${quote.text}"`, maxWidth);
    size -= 2;
  } while (lines.length * (size + 2) * 1.7 > 360 && size > 14);
  const lineHeight = (size + 2) * 1.7;
  const blockH = lines.length * lineHeight;
  let y = H / 2 - blockH / 2 - 20;
  ctx.fillStyle = "#f8fafc";
  ctx.textBaseline = "top";
  for (const line of lines) {
    ctx.fillText(line, left, y);
    y += lineHeight;
  }

  ctx.font = `18px ${family}`;
  ctx.fillStyle = accent;
  ctx.fillText(
    `- ${quote.author}${quote.source ? `, ${quote.source}` : ""}`,
    left,
    y + 18,
  );

  ctx.font = `14px ${family}`;
  ctx.fillStyle = "rgba(255,255,255,0.5)";
  ctx.fillText("QUOTE QUEST", W - 260, H - 76);

  const url = canvas.toDataURL("image/png");
  const a = document.createElement("a");
  a.href = url;
  a.download = `quote-${quote.author.toLowerCase().replace(/[^a-z0-9]+/g, "-")}.png`;
  a.click();
}
