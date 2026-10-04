"use client";

import { useState } from "react";

import { Badge } from "@/components/ui/8bit/badge";
import { Button } from "@/components/ui/8bit/button";
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/8bit/empty";
import { Label } from "@/components/ui/8bit/label";
import { ScrollArea } from "@/components/ui/8bit/scroll-area";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/8bit/sheet";
import { Switch } from "@/components/ui/8bit/switch";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/8bit/tabs";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/8bit/tooltip";
import { CATEGORY_BY_ID, type Quote } from "@/lib/quotes";

import { PixelSprite } from "./pixel-sprite";
import type { Settings } from "./use-quest-state";

interface InventorySheetProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  favorites: Quote[];
  log: Quote[];
  settings: Settings;
  stats: {
    xp: number;
    level: number;
    seen: number;
    total: number;
    forged: number;
  };
  onShow: (id: string) => void;
  onRemove: (id: string) => void;
  onSettings: (patch: Partial<Settings>) => void;
  onReset: () => void;
}

export function InventorySheet({
  open,
  onOpenChange,
  favorites,
  log,
  settings,
  stats,
  onShow,
  onRemove,
  onSettings,
  onReset,
}: InventorySheetProps) {
  const [confirmReset, setConfirmReset] = useState(false);

  return (
    <Sheet
      open={open}
      onOpenChange={(o) => {
        onOpenChange(o);
        setConfirmReset(false);
      }}
    >
      <div>
        <Tooltip>
          <TooltipTrigger>
            <SheetTrigger asChild>
              <Button variant="outline" className="h-9 gap-2 text-[10px]">
                <PixelSprite
                  name="scroll"
                  accent="#facc15"
                  className="size-4"
                />
                <span className="hidden sm:inline">INVENTORY</span>
                <span className="tabular-nums">{favorites.length}</span>
              </Button>
            </SheetTrigger>
          </TooltipTrigger>
          <TooltipContent>Saved quotes, log &amp; options (I)</TooltipContent>
        </Tooltip>
      </div>
      <SheetContent className="w-full gap-0 sm:max-w-md">
        <div className="flex h-full flex-col gap-4 p-5 pt-6">
          <SheetHeader className="p-0">
            <SheetTitle className="text-sm">INVENTORY</SheetTitle>
            <SheetDescription className="text-[10px]">
              LV {stats.level} &middot; {stats.xp} XP &middot; {stats.seen}/
              {stats.total} found
            </SheetDescription>
          </SheetHeader>

          <Tabs
            defaultValue="saved"
            className="flex min-h-0 flex-1 flex-col gap-4"
          >
            <TabsList className="w-full">
              <TabsTrigger value="saved" className="text-[10px]">
                SAVED
              </TabsTrigger>
              <TabsTrigger value="log" className="text-[10px]">
                LOG
              </TabsTrigger>
              <TabsTrigger value="options" className="text-[10px]">
                OPTIONS
              </TabsTrigger>
            </TabsList>

            <TabsContent value="saved" className="min-h-0 flex-1">
              {favorites.length === 0 ? (
                <EmptyState
                  title="NO ITEMS"
                  description="Press S or tap SAVE on a quote to stash it here."
                />
              ) : (
                <QuoteList
                  quotes={[...favorites].reverse()}
                  onShow={onShow}
                  action={(q) => (
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-7 px-2 text-[9px]"
                      onClick={() => onRemove(q.id)}
                      aria-label={`Remove quote by ${q.author}`}
                    >
                      DROP
                    </Button>
                  )}
                />
              )}
            </TabsContent>

            <TabsContent value="log" className="min-h-0 flex-1">
              {log.length === 0 ? (
                <EmptyState
                  title="EMPTY LOG"
                  description="Quotes you see this session appear here."
                />
              ) : (
                <QuoteList quotes={log} onShow={onShow} />
              )}
            </TabsContent>

            <TabsContent value="options" className="flex flex-col gap-6 pt-2">
              <Option
                id="opt-sound"
                label="SOUND FX"
                checked={settings.sound}
                onChange={(sound) => onSettings({ sound })}
              />
              <Option
                id="opt-crt"
                label="CRT SCANLINES"
                checked={settings.crt}
                onChange={(crt) => onSettings({ crt })}
              />
              <Option
                id="opt-type"
                label="TYPEWRITER TEXT"
                checked={settings.typewriter}
                onChange={(typewriter) => onSettings({ typewriter })}
              />
              <dl className="grid grid-cols-2 gap-3 text-[10px]">
                <dt className="text-muted-foreground">LEVEL</dt>
                <dd>{stats.level}</dd>
                <dt className="text-muted-foreground">TOTAL XP</dt>
                <dd>{stats.xp}</dd>
                <dt className="text-muted-foreground">DISCOVERED</dt>
                <dd>
                  {stats.seen}/{stats.total}
                </dd>
                <dt className="text-muted-foreground">FORGED</dt>
                <dd>{stats.forged}</dd>
              </dl>
              <Button
                variant="destructive"
                className="text-[10px]"
                onClick={() =>
                  confirmReset ? onReset() : setConfirmReset(true)
                }
              >
                {confirmReset ? "REALLY? CLICK AGAIN" : "RESET PROGRESS"}
              </Button>
            </TabsContent>
          </Tabs>
        </div>
      </SheetContent>
    </Sheet>
  );
}

function QuoteList({
  quotes,
  onShow,
  action,
}: {
  quotes: Quote[];
  onShow: (id: string) => void;
  action?: (q: Quote) => React.ReactNode;
}) {
  return (
    <ScrollArea className="h-[calc(100dvh-260px)] pr-3">
      <ul className="flex flex-col gap-4 py-1">
        {quotes.map((q, i) => {
          const meta = CATEGORY_BY_ID[q.category];
          return (
            <li
              key={`${q.id}-${i}`}
              className="flex gap-3 border-b-2 border-dashed border-border pb-4"
            >
              <PixelSprite
                name={q.custom ? "scroll" : q.category}
                accent={meta.color}
                className="mt-1 size-8 shrink-0"
              />
              <div className="flex min-w-0 flex-1 flex-col gap-2">
                <button
                  type="button"
                  onClick={() => onShow(q.id)}
                  className="line-clamp-3 cursor-pointer text-left text-[10px] leading-relaxed hover:underline"
                >
                  &ldquo;{q.text}&rdquo;
                </button>
                <div className="flex items-center justify-between gap-2">
                  <span className="truncate text-[9px] text-muted-foreground">
                    - {q.author}
                  </span>
                  <div className="flex items-center gap-2">
                    <Badge
                      className="text-[8px]"
                      style={{ background: meta.color, color: "#0b0b14" }}
                    >
                      {q.custom ? "FORGED" : meta.label.toUpperCase()}
                    </Badge>
                    {action?.(q)}
                  </div>
                </div>
              </div>
            </li>
          );
        })}
      </ul>
    </ScrollArea>
  );
}

function EmptyState({
  title,
  description,
}: {
  title: string;
  description: string;
}) {
  return (
    <Empty className="py-10">
      <EmptyHeader>
        <EmptyMedia>
          <PixelSprite
            name="gaming"
            accent="#64748b"
            className="size-14 animate-bob"
          />
        </EmptyMedia>
        <EmptyTitle className="text-xs">{title}</EmptyTitle>
        <EmptyDescription className="text-[10px] leading-relaxed">
          {description}
        </EmptyDescription>
      </EmptyHeader>
    </Empty>
  );
}

function Option({
  id,
  label,
  checked,
  onChange,
}: {
  id: string;
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <div className="flex items-center justify-between gap-4">
      <Label htmlFor={id} className="text-[10px]">
        {label}
      </Label>
      <Switch id={id} checked={checked} onCheckedChange={onChange} />
    </div>
  );
}
