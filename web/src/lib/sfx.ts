// Tiny chiptune sound engine: square/triangle oscillators, no audio files.

type Wave = OscillatorType;

interface Note {
  freq: number;
  /** Seconds from the start of the effect. */
  at: number;
  dur: number;
  wave?: Wave;
  gain?: number;
  /** Optional pitch the note slides to by the end. */
  slideTo?: number;
}

let ctx: AudioContext | null = null;

function audio() {
  if (typeof window === "undefined") return null;
  if (!ctx) {
    const Ctor =
      window.AudioContext ??
      (window as unknown as { webkitAudioContext?: typeof AudioContext })
        .webkitAudioContext;
    if (!Ctor) return null;
    ctx = new Ctor();
  }
  if (ctx.state === "suspended") void ctx.resume();
  return ctx;
}

function play(notes: Note[]) {
  const ac = audio();
  if (!ac) return;
  const t0 = ac.currentTime + 0.01;
  for (const n of notes) {
    const osc = ac.createOscillator();
    const amp = ac.createGain();
    osc.type = n.wave ?? "square";
    osc.frequency.setValueAtTime(n.freq, t0 + n.at);
    if (n.slideTo) {
      osc.frequency.exponentialRampToValueAtTime(n.slideTo, t0 + n.at + n.dur);
    }
    const peak = n.gain ?? 0.06;
    amp.gain.setValueAtTime(0, t0 + n.at);
    amp.gain.linearRampToValueAtTime(peak, t0 + n.at + 0.005);
    amp.gain.exponentialRampToValueAtTime(0.0001, t0 + n.at + n.dur);
    osc.connect(amp).connect(ac.destination);
    osc.start(t0 + n.at);
    osc.stop(t0 + n.at + n.dur + 0.02);
  }
}

const arp = (
  freqs: number[],
  step: number,
  dur: number,
  wave: Wave = "square",
  gain?: number,
) => freqs.map((freq, i) => ({ freq, at: i * step, dur, wave, gain }));

export const sfx = {
  /** Call from a user gesture so browsers allow playback. */
  unlock: () => void audio(),
  start: () => play(arp([523, 659, 784, 1047, 784, 1047], 0.08, 0.12)),
  coin: () =>
    play([
      { freq: 988, at: 0, dur: 0.07 },
      { freq: 1319, at: 0.07, dur: 0.25 },
    ]),
  blip: () =>
    play([{ freq: 1400 + Math.random() * 300, at: 0, dur: 0.025, gain: 0.02 }]),
  select: () =>
    play([
      { freq: 660, at: 0, dur: 0.06 },
      { freq: 880, at: 0.05, dur: 0.06 },
    ]),
  back: () =>
    play([
      { freq: 660, at: 0, dur: 0.06 },
      { freq: 440, at: 0.05, dur: 0.08 },
    ]),
  save: () => play(arp([392, 523, 659, 784], 0.05, 0.1, "triangle", 0.12)),
  unsave: () =>
    play([{ freq: 440, at: 0, dur: 0.25, slideTo: 110, gain: 0.05 }]),
  copy: () =>
    play([
      { freq: 1200, at: 0, dur: 0.05 },
      { freq: 1600, at: 0.06, dur: 0.08 },
    ]),
  levelUp: () =>
    play([
      ...arp([523, 659, 784, 1047], 0.07, 0.1),
      ...arp([587, 740, 880, 1175], 0.07, 0.1).map((n) => ({
        ...n,
        at: n.at + 0.32,
      })),
      { freq: 1319, at: 0.64, dur: 0.4 },
    ]),
  forge: () =>
    play([
      { freq: 200, at: 0, dur: 0.08, wave: "sawtooth", gain: 0.05 },
      { freq: 200, at: 0.12, dur: 0.08, wave: "sawtooth", gain: 0.05 },
      ...arp([523, 784, 1047], 0.06, 0.12).map((n) => ({
        ...n,
        at: n.at + 0.26,
      })),
    ]),
};

export type SfxName = keyof typeof sfx;
