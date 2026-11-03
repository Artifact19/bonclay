"use client";

import { useEffect } from "react";

// global-error replaces the root layout, so global CSS may not be present.
// Styles are inline with literal brand values on purpose.
export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("BonClay global error:", error);
  }, [error]);

  return (
    <html lang="en">
      <body
        style={{
          margin: 0,
          minHeight: "100vh",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "#0A0A0F",
          color: "#F8FAFC",
          fontFamily: "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif",
          padding: 16,
        }}
      >
        <div role="alert" style={{ maxWidth: 440, textAlign: "center" }}>
          <svg width="48" height="48" viewBox="0 0 32 32" aria-hidden style={{ marginBottom: 20 }}>
            <defs>
              <linearGradient id="g" x1="0" y1="0" x2="32" y2="32" gradientUnits="userSpaceOnUse">
                <stop stopColor="#F0A23A" />
                <stop offset="1" stopColor="#E0782A" />
              </linearGradient>
            </defs>
            <rect width="32" height="32" rx="9" fill="url(#g)" />
            {[
              [7, 12, 8],
              [11.5, 8, 16],
              [16, 5, 22],
              [20.5, 9, 14],
              [25, 13, 6],
            ].map(([x, y, h]) => (
              <rect key={x} x={x - 1.1} y={y + 1} width="2.2" height={h} rx="1.1" fill="#fff" />
            ))}
          </svg>
          <h1 style={{ fontSize: 24, fontWeight: 600, margin: "0 0 8px" }}>BonClay could not start</h1>
          <p style={{ fontSize: 14, lineHeight: 1.6, color: "#94A3B8", margin: "0 0 24px" }}>
            A critical error stopped the app from loading. Retrying usually fixes it. If it keeps happening, check your connection and reload the page.
          </p>
          <button
            type="button"
            onClick={() => reset()}
            style={{
              height: 40,
              padding: "0 20px",
              border: 0,
              borderRadius: 8,
              color: "#fff",
              fontSize: 14,
              fontWeight: 500,
              cursor: "pointer",
              background: "linear-gradient(135deg, #F0A23A 0%, #E0782A 100%)",
            }}
          >
            Retry
          </button>
        </div>
      </body>
    </html>
  );
}
