import React from "react";

/** BonClay' own navigation icons: 24px grid, 1.6 stroke, round caps, each built around the
 *  voice-bar motif from the logo. On the active item one detail lights up in marigold. */
export interface NavIconProps {
  className?: string;
  active?: boolean;
}

const MARIGOLD = "#F0A23A";

function Frame({ className, children }: { className?: string; children: React.ReactNode }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.6}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
      className={className}
    >
      {children}
    </svg>
  );
}

/** The lit detail follows the area accent (marigold in Studio, green in Agents); marigold when used outside an area. */
const accent = (active?: boolean) => (active ? `var(--area-accent, ${MARIGOLD})` : "currentColor");

/** Overview: three panels and a fourth that is speaking. */
export function OverviewIcon({ className, active }: NavIconProps) {
  return (
    <Frame className={className}>
      <rect x="3" y="3" width="7.5" height="7.5" rx="2" />
      <rect x="13.5" y="3" width="7.5" height="7.5" rx="2" />
      <rect x="3" y="13.5" width="7.5" height="7.5" rx="2" />
      <g stroke={accent(active)}>
        <path d="M14.5 16v3" />
        <path d="M17.25 14v7" />
        <path d="M20 15.5v4" />
      </g>
    </Frame>
  );
}

/** Voices: a head in profile, speaking. */
export function VoicesIcon({ className, active }: NavIconProps) {
  return (
    <Frame className={className}>
      <path d="M4.5 21v-3a7 7 0 1 1 11.4-5.4l1.3 2.5a.6.6 0 0 1-.5.9h-1.4v1.7a1.8 1.8 0 0 1-1.8 1.8H11.5V21" />
      <g stroke={accent(active)}>
        <path d="M19.2 9.2a4.2 4.2 0 0 1 0 4.6" />
        <path d="M21.5 7a7.6 7.6 0 0 1 0 9" />
      </g>
    </Frame>
  );
}

/** Studio: a waveform on a timeline with the playhead running through it. */
export function StudioIcon({ className, active }: NavIconProps) {
  return (
    <Frame className={className}>
      <path d="M3.5 10.5v3" />
      <path d="M6.5 8v8" />
      <path d="M9.5 5.5v13" />
      <path d="M15 9v6" />
      <path d="M18 6.5v11" />
      <path d="M21 10v4" />
      <g stroke={accent(active)}>
        <path d="M12.25 5.5V21" />
        <circle cx="12.25" cy="3.6" r="1.4" fill={accent(active)} />
      </g>
    </Frame>
  );
}

/** Agents: a speech bubble that talks back. */
export function AgentsIcon({ className, active }: NavIconProps) {
  return (
    <Frame className={className}>
      <path d="M4 6a3 3 0 0 1 3-3h10a3 3 0 0 1 3 3v7a3 3 0 0 1-3 3h-6l-4.5 4.2V16H7a3 3 0 0 1-3-3z" />
      <g stroke={accent(active)}>
        <path d="M9 8.5v2" />
        <path d="M12 7v5" />
        <path d="M15 8.5v2" />
      </g>
    </Frame>
  );
}

/** Say it once: one voice bar that splits into several, each one speaking. */
export function SayItOnceIcon({ className, active }: NavIconProps) {
  return (
    <Frame className={className}>
      <path d="M3.5 9.5v5" />
      <path d="M6 12c3 0 3-6.5 6.5-6.5" />
      <path d="M6 12h6.5" />
      <path d="M6 12c3 0 3 6.5 6.5 6.5" />
      <g stroke={accent(active)}>
        <path d="M15.5 3.5v4" />
        <path d="M19 4.5v2" />
        <path d="M15.5 10v4" />
        <path d="M19 10.75v2.5" />
        <path d="M15.5 16.5v4" />
        <path d="M19 17.5v2" />
      </g>
    </Frame>
  );
}

/** Playground: hold to talk, live. */
export function PlaygroundIcon({ className, active }: NavIconProps) {
  return (
    <Frame className={className}>
      <path d="M7.8 7.8a6 6 0 0 0 0 8.4" />
      <path d="M16.2 7.8a6 6 0 0 1 0 8.4" />
      <path d="M4.9 4.9a10 10 0 0 0 0 14.2" />
      <path d="M19.1 4.9a10 10 0 0 1 0 14.2" />
      <circle cx="12" cy="12" r="2.3" fill={accent(active)} stroke={accent(active)} />
    </Frame>
  );
}

/** Knowledge: a document that can be read aloud. */
export function KnowledgeIcon({ className, active }: NavIconProps) {
  return (
    <Frame className={className}>
      <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
      <path d="M14 3v5h5" />
      <g stroke={accent(active)}>
        <path d="M8.5 14.5v2" />
        <path d="M11 12v6.5" />
        <path d="M13.5 13.5v3.5" />
        <path d="M16 15v1" />
      </g>
    </Frame>
  );
}

/** Analytics: a trend line on a baseline, latest point marked. */
export function AnalyticsIcon({ className, active }: NavIconProps) {
  return (
    <Frame className={className}>
      <path d="M3 20.5h18" />
      <path d="M4 16.5l4.2-5 3.6 3 4.4-6 2.2 2" />
      <circle cx="19.6" cy="11.2" r="1.7" fill={accent(active)} stroke={accent(active)} />
    </Frame>
  );
}

/** Evaluations: crosshairs; every agent gets attacked before it ships. */
export function EvaluationsIcon({ className, active }: NavIconProps) {
  return (
    <Frame className={className}>
      <circle cx="12" cy="12" r="7" />
      <path d="M12 2.5v4" />
      <path d="M12 17.5v4" />
      <path d="M2.5 12h4" />
      <path d="M17.5 12h4" />
      <circle cx="12" cy="12" r="1.6" fill={accent(active)} stroke={accent(active)} />
    </Frame>
  );
}

/** API keys: code braces around a voice bar. */
export function ApiKeysIcon({ className, active }: NavIconProps) {
  return (
    <Frame className={className}>
      <path d="M8 4H7.2a2 2 0 0 0-2 2v3.4c0 1.3-.9 2.3-2.2 2.6 1.3.3 2.2 1.3 2.2 2.6V18a2 2 0 0 0 2 2H8" />
      <path d="M16 4h.8a2 2 0 0 1 2 2v3.4c0 1.3.9 2.3 2.2 2.6-1.3.3-2.2 1.3-2.2 2.6V18a2 2 0 0 1-2 2H16" />
      <g stroke={accent(active)}>
        <path d="M10 10.5v3" />
        <path d="M12 8.5v7" />
        <path d="M14 10.5v3" />
      </g>
    </Frame>
  );
}

/** Verify a clip: a shield with a voice bar and a check beneath it. */
export function VerifyIcon({ className, active }: NavIconProps) {
  return (
    <Frame className={className}>
      <path d="M12 3 4.5 5.8v5.6c0 4.4 3 7.6 7.5 9.6 4.5-2 7.5-5.2 7.5-9.6V5.8L12 3Z" />
      <g stroke={accent(active)}>
        <path d="m9 12.2 2.1 2.1L15.2 10" />
      </g>
    </Frame>
  );
}

/** Approvals: a calendar page with a tick; the HOD decides what gets booked. */
export function ApprovalsIcon({ className, active }: NavIconProps) {
  return (
    <Frame className={className}>
      <rect x="3.5" y="5" width="17" height="15" rx="2.5" />
      <path d="M3.5 10h17" />
      <path d="M8 3v4" />
      <path d="M16 3v4" />
      <path d="m9 15 2.2 2.2L15.5 13" stroke={accent(active)} />
    </Frame>
  );
}
