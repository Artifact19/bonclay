import { PageIntro, NextPage } from "@/components/site/PageIntro";
import { Consent, Pipeline } from "@/components/site/Sections";
import { pageMetadata } from "@/components/site/pageMeta";

export const metadata = pageMetadata({
  title: "How it works",
  description:
    "How a request becomes speech: the steps from your words to a watermarked clip, and the spoken consent that comes before any voice is cloned.",
  path: "/how-it-works",
});

export default function HowItWorksPage() {
  return (
    <>
      <PageIntro title="From a few words to a clip you can check.">
        The steps behind every reply, and the spoken yes that comes before any voice is cloned.
      </PageIntro>
      <Pipeline />
      <Consent />
      <NextPage from="/how-it-works" />
    </>
  );
}
