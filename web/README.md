# ML Lab web UI

A [Next.js](https://nextjs.org) app styled with [8bitcn](https://8bitcn.com), a retro 8-bit component library built on [shadcn/ui](https://ui.shadcn.com) and Tailwind CSS v4.

## Getting started

```bash
cd web
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The demo page is `src/app/page.tsx`.

## 8bitcn

Installed components live in `src/components/ui/8bit/`, with the shadcn base components they wrap in `src/components/ui/`:

alert, badge, button, card, dialog, health-bar, input, label, mana-bar, progress, separator, skeleton, tabs, xp-bar, plus `retro-mode-switcher` (light/dark toggle, backed by `src/components/theme-provider.tsx`).

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
