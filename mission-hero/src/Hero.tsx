import { useCallback, useEffect, useState, type CSSProperties } from 'react';
import { Example as BlackHole, type BlackHoleStatus } from './black-hole';

// If the GPU takes unusually long to compile, open the shutters anyway so the
// page never sits on a black screen.
const REVEAL_TIMEOUT_MS = 6000;

const TELEMETRY = [
  ['Target', 'Sagittarius A*'],
  ['Mass', '4.3 × 10⁶ M☉'],
  ['Distance', '26,673 ly'],
  ['Horizon radius', '12.7 M km'],
] as const;

export function Hero() {
  const [status, setStatus] = useState<BlackHoleStatus>('loading');
  const [revealed, setRevealed] = useState(false);

  const onStatusChange = useCallback((next: BlackHoleStatus, error?: unknown) => {
    setStatus(next);
    if (next === 'error') console.warn('[UMBRA-1] WebGPU unavailable, using still fallback.', error);
    if (next !== 'loading') setRevealed(true);
  }, []);

  useEffect(() => {
    const id = window.setTimeout(() => setRevealed(true), REVEAL_TIMEOUT_MS);
    return () => window.clearTimeout(id);
  }, []);

  return (
    <main className="hero" data-status={status} data-revealed={revealed}>
      {/* ——— The visual ——— */}
      <div
        className="hero__visual"
        role="img"
        aria-label="A black hole bending starlight around its event horizon, ringed by a glowing accretion disk."
      >
        <BlackHole className="hero__canvas" onStatusChange={onStatusChange} />
        <div className="hero__fallback" aria-hidden="true">
          <div className="fallback__disk" />
          <div className="fallback__ring" />
          <div className="fallback__core" />
        </div>
      </div>

      {/* ——— Atmosphere: scrims, grain, letterbox ——— */}
      <div className="hero__scrim" aria-hidden="true" />
      <div className="hero__grain" aria-hidden="true" />
      <div className="hero__bars" aria-hidden="true">
        <div className="bar bar--top" />
        <div className="bar bar--bottom" />
        <p className="hero__acquiring">
          <span className="pulse" /> Acquiring signal
        </p>
      </div>
      <div className="hero__frame" aria-hidden="true">
        <span /><span /><span /><span />
      </div>

      {/* ——— HUD ——— */}
      <header className="hud-top reveal" style={delay('0.2s')}>
        <a className="mark" href="#" aria-label="UMBRA-1 home">
          <svg viewBox="0 0 32 32" aria-hidden="true">
            <circle cx="16" cy="16" r="6.5" fill="#000" />
            <ellipse cx="16" cy="16" rx="14" ry="4.2" fill="none" stroke="currentColor" strokeWidth="1.4" />
            <circle cx="16" cy="16" r="9" fill="none" stroke="currentColor" strokeOpacity=".5" strokeWidth="1" />
          </svg>
          <span>
            UMBRA<b>—1</b>
          </span>
        </a>
        <nav aria-label="Primary">
          <a href="#mission">Mission</a>
          <a href="#science">Science</a>
          <a href="#spacecraft">Spacecraft</a>
          <a href="#crew">Crew</a>
        </nav>
        <p className="signal">
          <span className="pulse" /> NASA · Mission concept
        </p>
      </header>

      <aside className="coords reveal" style={delay('1.1s')} aria-hidden="true">
        RA 17ʰ45ᵐ40ˢ · DEC −29°00′28″ · GALACTIC CENTRE
      </aside>

      {/* ——— Copy ——— */}
      <section className="copy" aria-labelledby="hero-title">
        <p className="eyebrow reveal" style={delay('0.35s')}>
          <span className="eyebrow__rule" /> Mission 01 &nbsp;/&nbsp; Exploring a black hole
        </p>
        <h1 id="hero-title" className="title">
          <span className="line reveal" style={delay('0.5s')}>
            A mission into
          </span>
          <span className="line reveal" style={delay('0.68s')}>
            the <em>unknown.</em>
          </span>
        </h1>
        <p className="lede reveal" style={delay('0.9s')}>
          UMBRA-1 will travel farther than anything we have ever built — to the edge of a
          supermassive black hole, where gravity folds starlight into rings and time slows to a
          crawl. What it sends back will rewrite physics.
        </p>
        <div className="actions reveal" style={delay('1.05s')}>
          <a className="cta" href="#mission">
            <span>Begin the descent</span>
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M5 12h13M13 6l6 6-6 6" fill="none" stroke="currentColor" strokeWidth="1.6" />
            </svg>
          </a>
          <a className="cta-ghost" href="#science">
            Read the mission brief
          </a>
        </div>
      </section>

      {/* ——— Telemetry ——— */}
      <dl className="telemetry reveal" style={delay('1.2s')}>
        {TELEMETRY.map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
        <div>
          <dt>Mission clock</dt>
          <dd>
            <MissionClock />
          </dd>
        </div>
      </dl>

      <p className="hint reveal" style={delay('1.6s')}>
        {status === 'error' ? (
          'WebGPU unavailable — showing a still. Open in a WebGPU browser for the live simulation.'
        ) : (
          <>
            <span className="hint__dot" /> Move to orbit the horizon
          </>
        )}
      </p>
    </main>
  );
}

function MissionClock() {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const start = performance.now();
    const id = window.setInterval(() => setElapsed((performance.now() - start) / 1000), 1000);
    return () => window.clearInterval(id);
  }, []);
  const s = Math.floor(elapsed);
  const pad = (n: number) => String(n).padStart(2, '0');
  return (
    <time>
      T+ {pad(Math.floor(s / 3600))}:{pad(Math.floor(s / 60) % 60)}:{pad(s % 60)}
    </time>
  );
}

/** Staggered entrance delay, consumed by `.reveal` in styles.css. */
function delay(seconds: string): CSSProperties {
  return { '--d': seconds } as CSSProperties;
}
