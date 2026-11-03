import { PageIntro, NextPage } from "@/components/site/PageIntro";
import { Faq, FinalCta } from "@/components/site/Ending";
import { HOME } from "@/components/site/routes";
import { pageMetadata } from "@/components/site/pageMeta";

export const metadata = pageMetadata({
  title: "FAQ",
  description: "What BonClay costs, what runs on your own machine, which languages and hardware it needs, and what is not ready yet.",
  path: "/faq",
});

export default function FaqPage() {
  return (
    <>
      <PageIntro title="Questions people ask.">
        Open any question for the full answer.
      </PageIntro>
      <Faq />
      <FinalCta />
      <NextPage from="/faq" to={{ href: HOME.href, title: "Back to the start", blurb: HOME.blurb }} />
    </>
  );
}
