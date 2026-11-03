import Image from "next/image";
import styles from "./Logo.module.css";

/** Which ink to use. `auto` follows the app theme (and stays light on the marketing site). */
export type LogoTone = "auto" | "light" | "dark";

const ICON_SRC = "/brand/bonclay-icon.svg";

/** Brand spec: 64px wordmark sits beside an 80px icon on the desktop reference layout. */
const WORDMARK_PER_ICON = 64 / 80;
/** Below this icon size VOICE STUDIO would fall under ~9px and stop being legible, so it is hidden by default. */
const SUBTITLE_MIN_ICON = 64;

interface LogoMarkProps {
  /** Rendered width and height in px. */
  size?: number;
  /** Accessible name. Leave empty when the mark sits next to the wordmark or inside an already-labelled link. */
  label?: string;
  className?: string;
}

/** The BonClay icon (vector). Usable on its own. */
export function LogoMark({ size = 28, label = "", className = "" }: LogoMarkProps) {
  return (
    <Image
      src={ICON_SRC}
      width={size}
      height={size}
      alt={label}
      unoptimized
      loading="eager"
      className={`shrink-0 ${className}`}
    />
  );
}

interface WordmarkProps {
  /** Wordmark font-size in px; the 64px desktop reference scales every other proportion. */
  size?: number;
  /** Show VOICE STUDIO centred beneath the wordmark. */
  subtitle?: boolean;
  tone?: LogoTone;
  className?: string;
}

/** "BonClay" set in Manrope Regular as live text. Usable on its own. */
export function Wordmark({ size = 22, subtitle = false, tone = "auto", className = "" }: WordmarkProps) {
  return (
    <span className={`${styles.wordmark} ${styles[tone]} ${className}`} style={{ fontSize: size }}>
      <span className={styles.name}>BonClay</span>
      {subtitle && <span className={styles.subtitle}>VOICE STUDIO</span>}
    </span>
  );
}

interface LogoProps {
  /** Icon size in px; the wordmark is scaled from it (80px icon : 64px wordmark). */
  size?: number;
  className?: string;
  showText?: boolean;
  onlyText?: boolean;
  /** Force VOICE STUDIO on or off. By default it appears once the lockup is large enough to read it. */
  subtitle?: boolean;
  tone?: LogoTone;
}

/** Lockup: icon + wordmark, the text block vertically centred on the icon. */
export function Logo({ size = 32, className = "", showText = true, onlyText = false, subtitle, tone = "auto" }: LogoProps) {
  const withSubtitle = subtitle ?? size >= SUBTITLE_MIN_ICON;
  return (
    <span className={`${styles.logo} ${className}`} style={{ gap: Math.max(8, size * 0.3) }}>
      {!onlyText && <LogoMark size={size} />}
      {(showText || onlyText) && <Wordmark size={size * WORDMARK_PER_ICON} subtitle={withSubtitle} tone={tone} />}
    </span>
  );
}
