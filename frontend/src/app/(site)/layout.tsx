import { SmoothScroll } from "@/components/site/SmoothScroll";
import { SiteNav } from "@/components/site/SiteNav";
import { Footer } from "@/components/site/Footer";
import { siteFont, siteSerif } from "@/components/site/siteFont";
import { PublicMain } from "@/components/PublicMain";
import { manrope } from "@/lib/brandFont";

/**
 * Shared shell of the public marketing pages (/, /how-it-works, /channels, /proof, /run-it, /faq). It stays mounted
 * while a visitor moves between them, so the nav, the smooth scrolling and the footer are not rebuilt on every page.
 * PublicMain provides the skip link and <main id="main-content">, as the app shell does for the dashboard. The shared
 * skip link is redirected to #content by SmoothScroll so it lands after the nav. Manrope is here for the wordmark only;
 * Manrope and Instrument Serif come from siteFont. /verify, the app (dashboard, studio, voices, auth) and the rest of the root keep
 * their own layouts and load none of the dashboard code.
 */
export default function SiteLayout({ children }: { children: React.ReactNode }) {
  return (
    <PublicMain>
      <SmoothScroll>
        <div className={`site ${siteFont.variable} ${siteSerif.variable} ${manrope.variable} min-h-screen`}>
          <SiteNav />
          <div id="content">{children}</div>
          <Footer />
        </div>
      </SmoothScroll>
    </PublicMain>
  );
}
