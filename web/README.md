# Quote Quest

An 8-bit quote generator built with [Next.js](https://nextjs.org) and [8bitcn](https://8bitcn.com), a retro component library built on [shadcn/ui](https://ui.shadcn.com) and Tailwind CSS v4.

- 50 quotes across five categories (Wisdom, Code, Gaming, Science, Grit), each with a pixel-art character
- RPG-style dialogue box with typewriter text and chiptune sound effects (Web Audio, no audio files)
- XP, levels and a collection meter; favouring unseen quotes until you've found them all
- Inventory sheet with saved quotes, a session log, options and stats
- Copy, read aloud, share to X, or download a 1200x630 PNG quote card
- Forge your own quotes, which join the rotation
- Progress is kept in `localStorage`; full keyboard controls (press `SPACE` to start)

## Getting started

```bash
cd web
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The app lives in `src/components/quote-quest/`, quotes in `src/lib/quotes.ts`.

## 8bitcn

Installed components live in `src/components/ui/8bit/`, with the shadcn base components they wrap in `src/components/ui/`:

alert, avatar, badge, button, card, dialog, empty, health-bar, input, kbd, label, mana-bar, progress, scroll-area, select, separator, sheet, skeleton, spinner, switch, tabs, toast, toggle, toggle-group, tooltip, xp-bar, the `dialogue` block, plus `retro-mode-switcher` (light/dark toggle, backed by `src/components/theme-provider.tsx`).

Use them like any shadcn component:

```tsx
import { Button } from "@/components/ui/8bit/button";

<Button>Start</Button>;
```

Add the `retro` class to an element to use the Press Start 2P pixel font (see `src/components/ui/8bit/styles/retro.css`).

### Adding more components

`components.json` registers the `@8bitcn` registry, so the shadcn CLI can add more:

```bash
npx shadcn@latest add @8bitcn/accordion
```

Browse the full list at [8bitcn.com/docs/components](https://www.8bitcn.com/docs/components).
