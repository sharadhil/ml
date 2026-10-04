"use client";

import { useState } from "react";

import { Button } from "@/components/ui/8bit/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/8bit/dialog";
import { Input } from "@/components/ui/8bit/input";
import { Label } from "@/components/ui/8bit/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/8bit/select";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/8bit/tooltip";
import { CATEGORIES, type Category, type Quote } from "@/lib/quotes";

import { PixelSprite } from "./pixel-sprite";

const MAX_TEXT = 200;
const MAX_AUTHOR = 40;

interface ForgeDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onForge: (quote: Omit<Quote, "id" | "custom">) => void;
}

export function ForgeDialog({ open, onOpenChange, onForge }: ForgeDialogProps) {
  const [text, setText] = useState("");
  const [author, setAuthor] = useState("");
  const [category, setCategory] = useState<Category>("wisdom");
  const [error, setError] = useState("");

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const t = text.trim().replace(/^["“]|["”]$/g, "");
    if (t.length < 3) return setError("Your quote needs a few more words.");
    onForge({ text: t, author: author.trim() || "Anonymous", category });
    setText("");
    setAuthor("");
    setError("");
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <div>
        <Tooltip>
          <TooltipTrigger>
            <DialogTrigger asChild>
              <Button
                variant="outline"
                className="h-11 w-full gap-2 text-[10px]"
              >
                <PixelSprite
                  name="scroll"
                  accent="#facc15"
                  className="size-4"
                />
                FORGE
              </Button>
            </DialogTrigger>
          </TooltipTrigger>
          <TooltipContent>Write your own (F)</TooltipContent>
        </Tooltip>
      </div>
      <DialogContent className="sm:max-w-md">
        <form onSubmit={submit} className="flex flex-col gap-6">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-3 text-sm">
              <PixelSprite name="scroll" accent="#facc15" className="size-8" />
              FORGE A QUOTE
            </DialogTitle>
            <DialogDescription className="text-[10px] leading-relaxed">
              Add your own words to the collection. Forging grants bonus XP.
            </DialogDescription>
          </DialogHeader>

          <div className="flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <Label htmlFor="forge-text" className="text-[10px]">
                QUOTE
              </Label>
              <span className="text-[9px] text-muted-foreground">
                {text.length}/{MAX_TEXT}
              </span>
            </div>
            <Input
              id="forge-text"
              value={text}
              maxLength={MAX_TEXT}
              placeholder="Stay pixelated, my friend."
              onChange={(e) => {
                setText(e.target.value);
                setError("");
              }}
              autoComplete="off"
              aria-invalid={!!error}
              aria-describedby={error ? "forge-error" : undefined}
            />
            {error && (
              <p id="forge-error" className="text-[10px] text-destructive">
                {error}
              </p>
            )}
          </div>

          <div className="flex flex-col gap-3">
            <Label htmlFor="forge-author" className="text-[10px]">
              AUTHOR
            </Label>
            <Input
              id="forge-author"
              value={author}
              maxLength={MAX_AUTHOR}
              placeholder="Anonymous"
              onChange={(e) => setAuthor(e.target.value)}
              autoComplete="off"
            />
          </div>

          <div className="flex flex-col gap-3">
            <Label className="text-[10px]">CATEGORY</Label>
            <Select
              value={category}
              onValueChange={(v) => setCategory(v as Category)}
            >
              <SelectTrigger className="text-[10px]" aria-label="Category">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {CATEGORIES.map((c) => (
                  <SelectItem key={c.id} value={c.id} className="text-[10px]">
                    {c.label.toUpperCase()}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <DialogFooter>
            <Button type="submit" className="w-full">
              FORGE IT
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
