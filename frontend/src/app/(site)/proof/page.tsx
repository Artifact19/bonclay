import { PageIntro, NextPage } from "@/components/site/PageIntro";
import { Attacks } from "@/components/site/Sections";
import { Comparison, Measured } from "@/components/site/Proof";
import { WhySection } from "@/components/site/WhySection";
import { pageMetadata } from "@/components/site/pageMeta";

export const metadata = pageMetadata({
  title: "Proof",
  description:
    "What BonClay measured on one laptop, the attacks it was tested against, how it compares with hosted voice platforms, and the sourced numbers behind why it exists.",
  path: "/proof",
});

export default function ProofPage() {
  return (
    <>
      <PageIntro title="Numbers, attacks and comparisons.">
        What we measured on one laptop, what we threw at the agent, and how BonClay compares.
      </PageIntro>
      <Measured />
      <Attacks />
      <Comparison />
      {/* The full, sourced version of the short "Why" on Home (every figure links to where it came from). */}
      <WhySection id="why-evidence" />
      <NextPage from="/proof" />
    </>
  );
}
