// Deterministic so server and client render identical markup.
function seeded(seed: number) {
  return () => {
    seed = (seed * 16807) % 2147483647;
    return (seed - 1) / 2147483646;
  };
}

const rand = seeded(42);
const STARS = Array.from({ length: 70 }, (_, i) => ({
  left: rand() * 100,
  top: rand() * 78,
  size: i % 9 === 0 ? 4 : 2,
  delay: rand() * 4,
  duration: 2 + rand() * 3,
}));

export function Starfield() {
  return (
    <div className="pointer-events-none absolute inset-0 -z-10" aria-hidden>
      {STARS.map((s, i) => (
        <span
          key={i}
          className="star absolute"
          style={{
            left: `${s.left}%`,
            top: `${s.top}%`,
            width: s.size,
            height: s.size,
            animationDelay: `${s.delay}s`,
            animationDuration: `${s.duration}s`,
          }}
        />
      ))}
      <div className="moon absolute right-[8%] top-[10%] size-10 sm:size-14" />
      <div className="skyline absolute inset-x-0 bottom-0 h-16" />
    </div>
  );
}
