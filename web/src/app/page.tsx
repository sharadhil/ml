import { Badge } from "@/components/ui/8bit/badge";
import { Button } from "@/components/ui/8bit/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/8bit/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/8bit/dialog";
import HealthBar from "@/components/ui/8bit/health-bar";
import { Input } from "@/components/ui/8bit/input";
import { Label } from "@/components/ui/8bit/label";
import ManaBar from "@/components/ui/8bit/mana-bar";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/8bit/tabs";
import XpBar from "@/components/ui/8bit/xp-bar";
import { RetroModeSwitcher } from "@/components/ui/retro-mode-switcher";

const programs = [
  { file: "lp1.py", topic: "Find-S" },
  { file: "lp2_candidate.py", topic: "Candidate Elimination" },
  { file: "NaiveBayes.py", topic: "Naive Bayes" },
  { file: "lp8_kmeans_new.py", topic: "K-Means" },
  { file: "lp8_em_new.py", topic: "EM" },
  { file: "lp9_knn_new.py", topic: "KNN" },
];

export default function Home() {
  return (
    <main className="retro mx-auto flex w-full max-w-3xl flex-col gap-8 p-6">
      <header className="flex items-center justify-between gap-4">
        <h1 className="text-lg md:text-2xl">ML Lab</h1>
        <RetroModeSwitcher />
      </header>

      <Card>
        <CardHeader>
          <CardTitle>Player stats</CardTitle>
          <CardDescription>8bitcn components are installed.</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-4 text-xs">
          <span>HP</span>
          <HealthBar value={80} />
          <span>MP</span>
          <ManaBar value={45} />
          <span>XP</span>
          <XpBar value={65} />
        </CardContent>
      </Card>

      <Tabs defaultValue="programs">
        <TabsList>
          <TabsTrigger value="programs">Programs</TabsTrigger>
          <TabsTrigger value="run">Run</TabsTrigger>
        </TabsList>
        <TabsContent value="programs">
          <ul className="flex flex-col gap-3 p-4 text-xs">
            {programs.map((p) => (
              <li key={p.file} className="flex items-center justify-between gap-2">
                <span>{p.file}</span>
                <Badge>{p.topic}</Badge>
              </li>
            ))}
          </ul>
        </TabsContent>
        <TabsContent value="run">
          <Card>
            <CardContent className="flex flex-col gap-3 pt-6">
              <Label htmlFor="dataset">Dataset</Label>
              <Input id="dataset" placeholder="KNN-input.csv" />
            </CardContent>
            <CardFooter>
              <Dialog>
                <DialogTrigger asChild>
                  <Button>Start</Button>
                </DialogTrigger>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>Ready player one</DialogTitle>
                    <DialogDescription>
                      Hook this up to the Python scripts to run them from here.
                    </DialogDescription>
                  </DialogHeader>
                </DialogContent>
              </Dialog>
            </CardFooter>
          </Card>
        </TabsContent>
      </Tabs>
    </main>
  );
}
