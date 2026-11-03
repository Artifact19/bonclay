import { ImageResponse } from "next/og";

// The share card for links to the site (Open Graph and, through the same file, Twitter). Rendered at build time from
// this code only: no network fetches, and the default font bundled with next/og.
export const alt = "BonClay: your voice, in Indian languages";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

const INK = "#0E0F11";
const PAPER = "#F4F4F1";
const MARIGOLD = "#F0A23A";

// Forty bars with fixed heights from two sine waves, tapered at both ends. No randomness, so the image never changes
// between builds unless this code does.
const BARS = Array.from({ length: 40 }, (_, i) => {
  const wobble = 0.5 + 0.32 * Math.sin(i * 0.61) + 0.18 * Math.sin(i * 1.7 + 0.8);
  const level = Math.min(1, Math.max(0.08, wobble));
  const envelope = 0.35 + 0.65 * Math.sin((Math.PI * i) / 39);
  return Math.round(14 + level * envelope * 104);
});

export default function OpengraphImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          background: PAPER,
          color: INK,
          padding: "64px 72px",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div style={{ display: "flex", alignItems: "center", fontSize: 34, fontWeight: 700 }}>
            <div style={{ display: "flex", width: 16, height: 16, borderRadius: 8, background: MARIGOLD, marginRight: 14 }} />
            BonClay
          </div>
          <div style={{ display: "flex", fontSize: 24, opacity: 0.6 }}>bonclay.ayushchougula.in</div>
        </div>

        <div style={{ display: "flex", flexDirection: "column" }}>
          <div style={{ display: "flex", fontSize: 88, fontWeight: 700, lineHeight: 1.04, letterSpacing: -2 }}>
            Your voice, in Indian languages.
          </div>
          <div style={{ display: "flex", marginTop: 24, fontSize: 28, whiteSpace: "nowrap", opacity: 0.75 }}>
            22 Indian languages · consented cloning · every clip watermarked and signed
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", height: 120 }}>
          {BARS.map((h, i) => (
            <div key={i} style={{ display: "flex", width: 12, height: h, borderRadius: 6, background: MARIGOLD }} />
          ))}
        </div>
      </div>
    ),
    { ...size },
  );
}
