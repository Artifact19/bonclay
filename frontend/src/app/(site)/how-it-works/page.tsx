import dynamic from "next/dynamic";
import { PageIntro, NextPage } from "@/components/site/PageIntro";
import { pageMetadata } from "@/components/site/pageMeta";

const Pipeline = dynamic(() => import("@/components/site/PipelineJourney").then((m) => m.PipelineJourney), {
  ssr: true,
});

const Consent = dynamic(() => import("@/components/site/Consent").then((m) => m.Consent), {
  ssr: true,
});

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
        Follow every step from your spoken words to an authentic, digitally watermarked voice reply.
        Every response is verifiable, latency is measured across each pipeline stage, and every voice clone begins with explicit spoken consent.
        Explore the end-to-end journey from acoustic capture to cryptographic audio verification.
      </PageIntro>
      <Pipeline />
      <Consent />
      <NextPage from="/how-it-works" />
    </>
  );
}
