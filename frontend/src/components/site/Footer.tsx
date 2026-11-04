import Link from "next/link";
import { ArrowUpRight } from "lucide-react";
import { Logo } from "@/components/Logo";
import { MONUMENT_CREDITS } from "./monuments";
import { REPO_PUBLIC, REPO_URL } from "./repo";
import { ALL_SITE_PAGES } from "./routes";

const REPO = REPO_URL;

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type FooterLink = { label: string; href: string; external?: boolean; repo?: boolean };

const FOOTER_COLUMNS: { title: string; links: FooterLink[] }[] = [
  {
    title: "Explore",
    links: ALL_SITE_PAGES.map((p) => ({ label: p.nav, href: p.href })),
  },
  {
    title: "Product",
    links: [
      { label: "Studio", href: "/dashboard/studio" },
      { label: "Dashboard", href: "/dashboard" },
      { label: "Agents", href: "/dashboard/studio/agents" },
      { label: "Voices", href: "/dashboard/voices" },
      { label: "Verify a clip", href: "/verify" },
    ],
  },
  {
    title: "Developers",
    links: [
      { label: "Source on GitHub", href: REPO, external: true, repo: true },
      { label: "Website widget", href: `${REPO}/blob/main/docs/WIDGET.md`, external: true, repo: true },
      { label: "Phone and meetings", href: `${REPO}/blob/main/docs/PHONE_AND_MEETINGS.md`, external: true, repo: true },
      { label: "WhatsApp voice notes", href: `${REPO}/blob/main/docs/WHATSAPP.md`, external: true, repo: true },
      { label: "Receipt public key", href: `${API_BASE}/.well-known/bonclay-receipt-key`, external: true },
    ],
  },
  {
    title: "Project",
    links: [
      { label: "Status", href: `${REPO}/blob/main/STATUS.md`, external: true, repo: true },
      { label: "Measured results", href: `${REPO}/blob/main/docs/RESULTS.md`, external: true, repo: true },
      { label: "Licences", href: `${REPO}#-license--acknowledgments`, external: true, repo: true },
    ],
  },
];

const footLink =
  "inline-flex min-h-[2.75rem] items-center gap-1 s-body text-[var(--s-ink-2)] underline-offset-4 hover:text-[var(--s-ink)] hover:underline";

export function Footer() {
  return (
    <footer className="s-container mt-24 pb-12 md:mt-32" aria-label="Site footer">
      <div className="grid gap-12 border-t border-[var(--s-line)] pt-12 lg:grid-cols-[1fr_2.2fr]" data-reveal>
        <div>
          <Logo size={40} />
          <p className="mt-4 max-w-[36ch] s-body leading-relaxed text-[var(--s-mute)]">
            A voice studio for India.
          </p>
          <p className="mt-4 max-w-[36ch] s-small leading-relaxed text-[var(--s-mute)]">
            No trackers or analytics on this site. Cookies are used only to keep you signed in when a hosted setup is configured.
          </p>
        </div>
        <nav aria-label="Footer" className="grid grid-cols-2 gap-x-6 gap-y-8 sm:grid-cols-4">
          {FOOTER_COLUMNS.map((col) => ({ ...col, links: col.links.filter((l) => REPO_PUBLIC || !l.repo) })).filter((col) => col.links.length > 0).map((col) => (
            <div key={col.title} className={col.title === "Developers" ? "order-last col-span-2 sm:order-none sm:col-span-1" : ""}>
              <p className="s-small font-medium">{col.title}</p>
              <ul className="mt-2">
                {col.links.map((l) => (
                  <li key={l.label}>
                    {l.external ? (
                      <a href={l.href} className={footLink}>
                        {l.label}
                        <ArrowUpRight className="h-3.5 w-3.5 shrink-0" aria-hidden />
                      </a>
                    ) : l.href.startsWith("#") ? (
                      <a href={l.href} className={footLink}>
                        {l.label}
                      </a>
                    ) : (
                      <Link href={l.href} prefetch={false} className={footLink}>
                        {l.label}
                      </Link>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </nav>
      </div>

      <div className="mt-12 grid gap-10 border-t border-[var(--s-line)] pt-10 md:grid-cols-[1fr_1.4fr]" data-reveal>
        <div>
          <p className="s-small font-medium">Open models</p>
          <ul className="mt-3 space-y-1.5 s-body text-[var(--s-mute)]">
            <li>Kokoro-82M</li>
            <li>Qwen3-TTS 0.6B</li>
            <li>DhVaani by ARTPARK-IISc</li>
            <li>Chatterbox Multilingual</li>
            <li>Indic-Speak by Bodhan AI / AI4Bharat</li>
            <li>Whisper and AudioSeal</li>
            <li>Llama 3.3 70B and Llama 3.2 3B</li>
          </ul>
        </div>
        <div>
          <p className="s-small font-medium">Notices</p>
          <p className="mt-3 max-w-[40ch] s-small leading-relaxed text-[var(--s-mute)]">
            Built with Indic-Speak from Bodhan AI / AI4Bharat.
            <br />
            Voice cloning in Indian languages by DhVaani from ARTPARK-IISc.
            <br />
            Built with Llama.
          </p>
          <p className="mt-3 max-w-[40ch] s-small leading-relaxed text-[var(--s-mute)]">
            BonClay is an educational and research project. Each model keeps its own licence{REPO_PUBLIC ? "; see the project licences linked above." : ": the models and their authors are listed above."}
          </p>
          <details className="mt-3 max-w-[44ch] s-small leading-relaxed text-[var(--s-mute)]">
            <summary className="inline-flex min-h-[1.5rem] cursor-pointer items-center underline underline-offset-4 hover:text-[var(--s-ink)]">
              Photo credits
            </summary>
            <p className="mt-2">Monument photos on the language cards are from Wikimedia Commons, scaled and slightly desaturated.</p>
            <ul className="mt-2 space-y-1">
              {MONUMENT_CREDITS.map((c) => (
                <li key={c.code}>
                  <a href={c.source} className="underline underline-offset-2 hover:text-[var(--s-ink)]">
                    {c.name}
                  </a>
                  {", "}
                  {c.author},{" "}
                  <a href={c.licenseUrl} className="underline underline-offset-2 hover:text-[var(--s-ink)]">
                    {c.license}
                  </a>
                </li>
              ))}
            </ul>
          </details>
        </div>
      </div>
    </footer>
  );
}
