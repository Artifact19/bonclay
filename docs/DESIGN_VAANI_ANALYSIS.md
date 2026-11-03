# vaani.dev design system, taken apart, and what BonClay takes from it

Measured in the browser on 30 October 2026 (computed styles, stylesheet rules and keyframes on https://www.vaani.dev/),
not guessed from screenshots. The second half maps each finding onto the BonClay marketing site and says what was kept
from ours, what was borrowed, and what was left out.

## 1. Colour

Vaani is a warm paper site. Nothing on it is pure white or pure black; every neutral leans amber.

| Role | Value | Where it shows |
| --- | --- | --- |
| Paper (ground) | `#F7F4ED` (`--vaani-paper`) | the page, every section background |
| Surface | `#EFEBE3` | the band behind the product console |
| Hero wash | `#F6F1E8` | hero panel behind the painting |
| Card / console face | `#FBFAF7` | the white-ish product mock, buttons on dark |
| Ink | `#292C28` (`--vaani-ink`), `#1A1F24` (`--nv-fg`, the h1) | headings, body on paper |
| Muted | `#6E5A49` (`--vaani-muted`), `#625E52` | paragraphs, captions (a brown, never a grey) |
| Hairline | `#E6DFD4` (`--nv-border`) | card borders, section rules |
| Bronze accent | `#927751` (hero italic line), `#8C6B45` / `#B57940` (`--nv-accent`) | the italic phrase in every headline |
| Night | `#15110E` (`--vaani-night`) | stats band, feature cards, closing banner |
| Gold (proof) | `#F2C78F` (`--vaani-proof`) | the big numerals on night |
| Night text | `#F4EBDE` | copy on night panels |
| Brand gradient | `linear-gradient(115deg, #2A3038 0%, #1A1F24 55%, #0F1216 100%)` | primary buttons, "brand" text fill |
| Live red | `#D94E3F` | the one hot colour, on the "LIVE TODAY" dot only |

Rules they follow: one accent (bronze) used only on type, dark panels carry the drama, and the warm neutral is used
at three depths (paper, surface, face) instead of grey tints.

## 2. Type

Two families, each with one job.

- Display: **Instrument Serif** 400, normal and italic. h1 `clamp(48px, 6.6vw, 96px)`, measured 67.6px at 1024 wide,
  tracking `-0.04em`, line height `1.02`. h2 `clamp(36px, 4.2vw, 60px)`, tracking `-0.03em`, line height `1.06`. The
  second line (or a phrase) of every heading is italic in bronze. Quote-like card headings use the serif at 28px.
- Text and UI: **Manrope** variable. Body 16 to 17px at 400, line height 1.65 to 1.7, muted brown, measure capped at
  610 to 700px. Card titles 18 to 20px at 500 to 600, tracking `-0.01em`. Buttons 12 to 14px at 600. Eyebrows 10 to
  12px at 600 to 700, uppercase, tracking `0.12em`. Big numerals Manrope 500 `clamp(2.8rem, 4.6vw, 3.6rem)`, tracking
  `-0.03em`, line height 1.
- Scale tokens: `--t-display`, `--t-h2`, `--t-h3 clamp(28px,2.5vw,36px)`, `--t-h4 clamp(22px,1.8vw,26px)`, `--t-title
  18px`, `--t-lead clamp(16px,1.2vw,17px)`, `--t-body 16px`, `--t-card 14px`, `--t-small 13px`, `--t-label 12px`.
- Indic text falls back to the Noto Serif family per script, listed in the serif stack.

## 3. Layout and spacing

- Max width `1440px`, gutter `clamp(22px, 4vw, 64px)`, section gap `clamp(64px, 8.75vw, 112px)`, heading to copy
  `clamp(28px, 3.75vw, 48px)`, copy gap 16px, surface gap 24px.
- Sections are full-width paper; the hero and the big dark panels are rounded **islands** inset by the gutter (24px
  radius) rather than full-bleed bands.
- A dot grid sits behind product mocks: `radial-gradient(circle, #2B211929 .8px, transparent 1.1px)` at `24px 24px`,
  drawn on a `::before` that spans `100vw`.
- Long stories (Teacher AI, Memories) are a left column of steps with a **sticky console** on the right.

## 4. Components

- **Nav**: fixed, transparent at the top. On scroll the whole bar pops into a centred white pill (radius 9999, hairline
  border, soft shadow) with a `header-pop-in` keyframe: `translateY(-100% - 12px) scale(.94)` to a 5px overshoot at 72%
  then rest, 0.42s. Links 15px 500; "Contact" is an outlined pill; "Try the platform" is the dark gradient pill, 14px
  600, 8px 18px padding.
- **Primary button**: dark brand gradient, pill, 13px 600 Manrope, `13px 22px` padding, a small circular arrow chip at
  the right, shadow `0 8px 18px rgba(79,59,36,.06)`, and a shimmer sweep (`nv-shimmer-slide`, 2.4s) on hover.
- **Secondary button**: paper at 75% opacity with hairline, same shape, same shadow.
- **"Stacked paper" card shadow** (their nicest detail): `inset 0 1px 0 #fff, 0 0 0 1px rgba(43,33,25,.08), 0 5px 0
  -1px #EFE7DA, 0 8px 18px rgba(90,60,30,.05)`: a white top highlight, a hairline, a cream lip underneath, a soft drop.
- **Night panel** (`.vaani-gradient`): `#15110E` plus `radial-gradient(ellipse at 80% 10%, #B17C4938, transparent 65%)`
  plus a charcoal texture image, `::before` with two more radial glows drifting 16s alternate, border
  `rgba(247,244,237,.19)`, shadow `0 2px 3px rgba(23,16,10,.07), 0 24px 48px -36px rgba(51,35,21,.44)`, radius 24px.
- **Stats band**: a night panel with three gold numerals (`dd`) above 15px labels (`dt`), centred, `column-reverse`.
- **Logo marquee**: two rows, 95s and 105s linear, opposite directions, logos grayscale at 75% opacity, edges masked
  `linear-gradient(90deg, transparent, #000 8%, #000 92%, transparent)`.
- **Feature cards** (dead air, turn taking...): night, 4px radius, 28px padding, chat bubbles inside, a small
  soundwave (`nv-soundwave-pulse`, `nv-eq-bounce` scaleY bars).
- **Case study card**: night with a bronze eyebrow, serif quote heading, three numerals, a small progress trace, arrows
  for the carousel.
- **Closing**: a cloudy-sky photo behind a serif headline whose italic line is white with a shadow, two pill buttons,
  then "backed by" logos and a four-column footer.
- Radii seen, in order of frequency: 9999 (pills), 16, 24, 22, 12, 10, 8, 4.

## 5. Motion

- Easing: `cubic-bezier(.22,1,.36,1)` and `cubic-bezier(.16,1,.3,1)`; no bounce except the nav pop.
- Entrances: `nv-plat-in-up` 44px, `nv-console-in-right` 56px, `memory-pop` 40px + scale .94, 0.5 to 0.95s.
- Ambient: gradient drift (`translate -5% -4%` to `5% 4%`, 16s alternate) on every dark panel; shimmer sweeps on
  buttons; soundwave bars; a caret blink and typing dots in the Teacher AI console; the hero is a pinned two-stage scene
  that crossfades the headline.
- Everything ambient is paused under `prefers-reduced-motion`.

## 6. What BonClay already does well (kept)

- The **monument language cards**: a photo of the state capital's monument behind the greeting, a frosted panel with
  the sentence and the language name. Vaani has nothing like it; it is the most Indian thing on either site.
- The **voice reel** in the hero (a playable list with a live waveform). It is proof, not decoration.
- The **facts row** of verifiable claims instead of customer logos.
- **Products** as two big panels, one night and one marigold, with the real product screenshots sliding up on hover.
- **Hear five**: playable clips with the engine credit on every card.
- Marigold `#F0A23A` as the brand colour, the consent story, the honest small print.

## 7. What BonClay takes from Vaani (changed)

| Vaani | BonClay now |
| --- | --- |
| Paper `#F7F4ED`, brown muted, hairline `#E6DFD4` | ground, muted and hairline tokens moved to the warm set; the cool grey `#F4F4F1` is gone |
| Instrument Serif headings with an italic bronze phrase | `.display`, `.h2`, `.s-title`, `.s-h3` are Instrument Serif; `.accent` is the italic bronze phrase |
| Manrope for everything else | Manrope replaces Anek Latin for body and UI (the Anek Indic cuts stay for the scripts) |
| Hero as a rounded island with a soft wash | hero panel: paper island, dot grid, amber glows, concentric rings, centred headline, two pill buttons, facts pill |
| Night stats band with gold numerals | new band under the hero: 22 languages, 0.84 s to first audio, 1 laptop |
| Dark gradient pill button with arrow chip and shimmer | `.s-btn--primary` |
| Stacked-paper shadow | `--s-shadow-paper` on cards |
| Logo marquee, slow, masked edges | the stack strip is a 70s marquee |
| Dot grid behind the console | dot grid behind the Products panels |
| Night panel with drifting amber glow | `.s-night-panel` on the Studio product panel and the closing section |
| Eyebrow labels | `.s-eyebrow` above section headings |
| Floating pill nav | the nav becomes a centred pill after the first scroll |

Left out on purpose: the renaissance painting (not ours to use), the pinned two-stage hero (it hid the voice reel), the
Product Hunt banner, the red live dot, the sticky-console stories (no content for them yet).

## 8. The language carousel

Asked for: a slow horizontal carousel that does not depend on scrolling. The cards now sit in a track that is
duplicated once and moves with one CSS animation (`translateX(-50%)`, 80s, linear, infinite), so the loop is seamless
and costs no JavaScript per frame. It pauses while the pointer or keyboard focus is over it, and under
`prefers-reduced-motion` it is a plain native scroller with the arrow buttons.
