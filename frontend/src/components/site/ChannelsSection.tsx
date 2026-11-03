"use client";

/* "One agent, every channel": a sticky-rail section. On wide screens with motion allowed the stage sticks
 * (CSS position: sticky, so no pin spacer is inserted and nothing can shift layout) while a ScrollTrigger
 * with snap swaps the active channel. Phones, short windows and reduced motion get a plain stacked list. */

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { ArrowUpRight } from "lucide-react";
import { useLenis } from "./SmoothScroll";
import { CHANNELS, DOCS_BASE, type Channel } from "./channelsData";
import { REPO_PUBLIC } from "./repo";
import { MeetingsMock, PhoneMock, WhatsAppMock, WidgetMock } from "./ChannelMocks";

gsap.registerPlugin(ScrollTrigger);

/** The same condition drives the CSS below and gsap.matchMedia, so markup and behaviour never disagree. */
const PIN_MQ = "(min-width: 1024px) and (min-height: 720px) and (prefers-reduced-motion: no-preference)";
const STAGE_TOP = 80; // px, only a fallback: the fixed nav is 4 rem tall and the CSS below rests the stage at 5 rem (its computed top is read at run time)
const N = CHANNELS.length;

const CSS = `
.chn-rail{display:none}
.chn-panels{display:grid;grid-template-columns:minmax(0,1fr);gap:4rem}
.chn-panel{display:grid;grid-template-columns:minmax(0,1fr);gap:1.25rem}
.chn-block{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:0.875rem;border-radius:var(--s-r-card);background:var(--s-marigold);color:var(--s-marigold-ink);padding:1.25rem 0.875rem}
.chn-eqbar{transform-origin:center;animation:none}
.chn-panel[data-play="true"] .chn-eqbar{animation:eq 0.9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.chn-panel[data-play="true"] .chn-eqbar{animation:none}}
@media (min-width:640px){.chn-block{padding:2rem 1.75rem}}
@media (min-width:768px){
  .chn-panel{grid-template-columns:minmax(0,.8fr) minmax(0,1.2fr);column-gap:2.5rem}
  .chn-head{grid-column:1;grid-row:1 / span 2}
  .chn-block{grid-column:2;grid-row:1}
  .chn-foot{grid-column:2;grid-row:2}
}
@media ${PIN_MQ}{
  .chn{--chn-h:min(40rem,calc(100svh - 6rem))}
  .chn-track{height:calc(var(--chn-h) + ${(N - 1) * 85}svh)}
  .chn-stage{position:sticky;top:max(5rem,calc((100svh - var(--chn-h)) / 2));height:var(--chn-h)}
  .chn-grid{display:grid;grid-template-columns:21.25rem minmax(0,1fr);gap:3rem;height:100%}
  .chn-rail{display:flex;flex-direction:column;justify-content:center}
  .chn-panels{display:block;position:relative;height:100%}
  .chn-panel{position:absolute;inset:0;display:flex;flex-direction:column;gap:1rem;opacity:0;visibility:hidden;transition:opacity .4s ease,visibility 0s .4s}
  .chn-panel[data-active="true"]{opacity:1;visibility:visible;transition:opacity .4s ease}
  .chn-head{display:none}
  .chn-block{flex:1;min-height:0;overflow:hidden;padding:1.5rem 1.75rem}
}
.chn-rail-line{position:absolute;left:0;top:0;bottom:0;width:1px;background:var(--s-line)}
.chn-rail-fill{position:absolute;left:0;top:0;bottom:0;width:1px;background:var(--s-ink);transform-origin:top;transform:scaleY(var(--p,0))}
.chn-rail-btn{display:grid;grid-template-columns:2.25rem 1fr;width:100%;text-align:left;padding:0.875rem 0 0.875rem 1.125rem;min-height:2.75rem;color:var(--s-mute);cursor:pointer;background:none;border:0;border-radius:0}
.chn-rail-btn:hover{color:var(--s-ink)}
.chn-rail-btn[aria-current="true"]{color:var(--s-ink)}
.chn-rail-btn[aria-current="true"] .chn-n{color:var(--s-ink)}
.chn-rail-btn[aria-current="true"]::before{content:"";position:absolute;left:-1px;width:3px;top:0.625rem;bottom:0.625rem;background:var(--s-marigold);border-radius:2px}
.chn-rail-item{position:relative}
.chn-desc{display:grid;margin:0.875rem 0 0 1.125rem;padding-left:2.25rem}
.chn-desc>p{grid-area:1/1;opacity:0;visibility:hidden;transition:opacity .3s ease,visibility 0s .3s}
.chn-desc>p[data-on="true"]{opacity:1;visibility:visible;transition:opacity .3s ease}
@media (prefers-reduced-motion:reduce){.chn-desc>p{transition:none}}
`;

function StatusLine({ c }: { c: Channel }) {
  const beta = c.tone === "beta";
  return (
    <footer data-reveal className="chn-foot flex flex-col gap-x-6 gap-y-2 sm:flex-row sm:items-start sm:justify-between">
      <p className="flex max-w-[58ch] items-start gap-2.5 s-small leading-[1.5] text-[var(--s-ink-2)]">
        <span
          aria-hidden
          className={`mt-[0.3125rem] h-2.5 w-2.5 shrink-0 rounded-full ${beta ? "border-2 border-[var(--s-ink)]" : c.tone === "tests" ? "border-2 border-[var(--s-ink)] bg-[var(--s-marigold)]" : "bg-[var(--s-ink)]"}`}
        />
        <span>
          <span className="font-semibold text-[var(--s-ink)]">{c.status}.</span> {c.statusDetail}
        </span>
      </p>
      {REPO_PUBLIC ? (
        <a
          href={DOCS_BASE + c.docPath}
          target="_blank"
          rel="noreferrer"
          className="inline-flex min-h-[2.75rem] shrink-0 items-center gap-1 s-small font-medium text-[var(--s-ink)] underline decoration-[var(--s-bar)] underline-offset-4 hover:decoration-[var(--s-ink)]"
        >
          {c.docLabel} on GitHub <ArrowUpRight size={15} aria-hidden />
        </a>
      ) : (
        <span className="inline-flex min-h-[2.75rem] shrink-0 items-center s-small text-[var(--s-mute)]">{c.docLabel}: shared on request</span>
      )}
    </footer>
  );
}

function Mock({ id, play }: { id: Channel["id"]; play: boolean }) {
  const mocks = {
    phone: <PhoneMock play={play} />,
    meetings: <MeetingsMock play={play} />,
    website: <WidgetMock play={play} />,
    whatsapp: <WhatsAppMock play={play} />,
  } as const;
  return mocks[id];
}

const LABEL: Record<Channel["id"], string> = {
  phone: "Example conversation, not a real caller",
  meetings: "Example, made-up names and times",
  website: "Example page",
  whatsapp: "Example conversation, not a real customer",
};

export function Channels() {
  const track = useRef<HTMLDivElement>(null);
  const stage = useRef<HTMLDivElement>(null);
  const rail = useRef<HTMLElement>(null);
  const trigger = useRef<ScrollTrigger | null>(null);
  const lock = useRef(false);
  const lenis = useLenis();
  const [active, setActive] = useState(0);
  const [pinned, setPinned] = useState(false);
  const [stageSeen, setStageSeen] = useState(false);
  const [seen, setSeen] = useState<boolean[]>(() => CHANNELS.map(() => false));

  useLayoutEffect(() => {
    const mm = gsap.matchMedia();
    mm.add(PIN_MQ, () => {
      setPinned(true);
      // Where the sticky stage actually rests (CSS decides), so the trigger range matches it exactly.
      const top = () => parseFloat(getComputedStyle(stage.current!).top) || STAGE_TOP;
      const st = ScrollTrigger.create({
        trigger: track.current,
        start: () => `top top+=${top()}`,
        end: () => `bottom top+=${top() + stage.current!.offsetHeight}`,
        invalidateOnRefresh: true,
        snap: { snapTo: 1 / (N - 1), duration: { min: 0.2, max: 0.5 }, delay: 0.1, ease: "power2.inOut" },
        onUpdate: (self) => {
          rail.current?.style.setProperty("--p", String(self.progress));
          if (!lock.current) setActive(Math.round(self.progress * (N - 1)));
        },
      });
      trigger.current = st;
      return () => {
        st.kill();
        trigger.current = null;
        setPinned(false);
        setActive(0);
      };
    });
    return () => mm.revert();
  }, []);

  // Which panels are on screen: the stage while pinned, each card when stacked.
  useEffect(() => {
    if (typeof IntersectionObserver === "undefined") {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- no observer: show everything played
      setStageSeen(true);
      setSeen(CHANNELS.map(() => true));
      return;
    }
    const stageIo = new IntersectionObserver(([e]) => setStageSeen(e.isIntersecting), { threshold: 0.35 });
    if (stage.current) stageIo.observe(stage.current);
    const cardIo = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (!e.isIntersecting) continue;
          const i = Number((e.target as HTMLElement).dataset.i);
          setSeen((s) => (s[i] ? s : s.map((v, k) => (k === i ? true : v))));
        }
      },
      { threshold: 0.3 }
    );
    track.current?.querySelectorAll("article[data-i]").forEach((el) => cardIo.observe(el));
    return () => {
      stageIo.disconnect();
      cardIo.disconnect();
    };
  }, []);

  const goTo = (i: number) => {
    const st = trigger.current;
    setActive(i);
    if (!st) return;
    const y = st.start + (st.end - st.start) * (i / (N - 1));
    lock.current = true;
    const unlock = () => {
      lock.current = false;
    };
    window.setTimeout(unlock, 1100);
    if (lenis) lenis.scrollTo(y, { duration: 0.9, onComplete: unlock });
    else window.scrollTo({ top: y, behavior: "smooth" });
  };

  return (
    <section id="channels" aria-labelledby="channels-h" className="chn mt-28 md:mt-36">
      <style>{CSS}</style>
      <div className="s-container">
        <h2 id="channels-h" data-reveal className="h2 max-w-[18ch]">
          One agent, every channel.
        </h2>
        <p data-reveal className="lede mt-5">
          Set your agent up once, with your own documents. Then people reach the same agent by phone, in an office, on your website and on WhatsApp. Each channel below says how far it has been tested.
        </p>
      </div>

      <div ref={track} className="chn-track mt-12 md:mt-16">
        <div ref={stage} className="s-container chn-stage">
          <div className="chn-grid">
            <nav ref={rail} className="chn-rail relative" aria-label="Channels">
              <span className="chn-rail-line" aria-hidden />
              <span className="chn-rail-fill" aria-hidden />
              <ul>
                {CHANNELS.map((c, i) => (
                  <li key={c.id} className="chn-rail-item">
                    <button type="button" className="chn-rail-btn" aria-current={active === i} aria-controls={`channel-${c.id}`} onClick={() => goTo(i)}>
                      <span className="chn-n pt-1 s-label font-semibold tabular-nums text-[var(--s-mute)]">{c.n}</span>
                      <span className="block">
                        <span className="s-item block font-medium text-[var(--s-ink)]">{c.title}</span>
                        <span className="mt-0.5 block s-small leading-5 text-[var(--s-mute)]">{c.short}</span>
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
              {/* All four descriptions share one grid cell, so the rail never changes height while you scroll. */}
              <div className="chn-desc" id="channels-desc">
                {CHANNELS.map((c, i) => (
                  <p key={c.id} data-on={active === i} className="max-w-[34ch] s-small leading-[1.55] text-[var(--s-ink-2)]">
                    <span className="mb-1 block s-label font-semibold text-[var(--s-mute)]">{c.title}</span>
                    {c.desc}
                  </p>
                ))}
              </div>
            </nav>

            <div className="chn-panels">
              {CHANNELS.map((c, i) => {
                const play = pinned ? stageSeen && active === i : seen[i];
                const hiddenNow = pinned && active !== i;
                return (
                  <article
                    key={c.id}
                    id={`channel-${c.id}`}
                    data-i={i}
                    data-active={active === i}
                    data-play={play}
                    aria-labelledby={`channel-${c.id}-t`}
                    inert={hiddenNow}
                    className="chn-panel"
                  >
                    <div data-reveal className="chn-head">
                      <p className="s-label font-semibold tabular-nums text-[var(--s-mute)]">{c.n}</p>
                      <h3 id={`channel-${c.id}-t`} className="s-h3 mt-1">
                        {c.title}
                      </h3>
                      <p className="mt-3 max-w-[44ch] s-body leading-[1.55] text-[var(--s-ink-2)]">{c.desc}</p>
                    </div>
                    <div data-reveal="media" className="chn-block">
                      <p className="rounded-xl bg-[var(--s-ink)] px-3 py-1 text-center s-label font-semibold leading-[1.5] text-[var(--s-night-text)] sm:rounded-full">{LABEL[c.id]}</p>
                      <Mock id={c.id} play={play} />
                    </div>
                    <StatusLine c={c} />
                  </article>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
