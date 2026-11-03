const HUES = ["#F0A23A", "#E0782A", "#C9A227", "#8C6B45", "#B5651D", "#D98E4A"];
const BARS = 7;

/** A simple 32-bit string hash (FNV-1a), so the same name always gives the same mark. */
function hash(name: string): number {
  let h = 2166136261;
  for (let i = 0; i < name.length; i++) {
    h ^= name.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

/** A deterministic "voiceprint" for an agent: a warm hue and seven bars, both derived from its name. */
export function VoiceMark({
  name,
  size = 40,
  className,
}: {
  name: string;
  size?: number;
  className?: string;
}) {
  const h = hash(name);
  const hue = HUES[h % HUES.length];
  const barW = 2.6;
  const span = 40 - 2 * 7; // usable width inside a 7 unit margin
  const gap = (span - BARS * barW) / (BARS - 1);
  const bars = Array.from({ length: BARS }, (_, i) => {
    // 25%..100% of the usable height (26 units), from successive bytes of a re-mixed hash
    const mixed = Math.imul(h ^ (i * 0x9e3779b1), 2246822519) >>> 0;
    const pct = 0.25 + ((mixed >>> 8) % 76) / 100;
    const height = pct * 26;
    return { x: 7 + i * (barW + gap), height, y: (40 - height) / 2 };
  });
  return (
    <svg
      viewBox="0 0 40 40"
      width={size}
      height={size}
      aria-hidden="true"
      focusable="false"
      className={className}
      style={{ flexShrink: 0 }}
    >
      <rect width="40" height="40" rx="10" fill={hue} fillOpacity="0.15" />
      {bars.map((b, i) => (
        <rect
          key={i}
          x={b.x}
          y={b.y}
          width={barW}
          height={b.height}
          rx={barW / 2}
          fill={hue}
        />
      ))}
    </svg>
  );
}
