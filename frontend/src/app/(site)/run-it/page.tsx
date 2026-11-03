import { PageIntro, NextPage } from "@/components/site/PageIntro";
import { RigCheck } from "@/components/site/RigCheck";
import { pageMetadata } from "@/components/site/pageMeta";

export const metadata = pageMetadata({
  title: "Can you run it?",
  description:
    "Pick your laptop and see what BonClay does well on it, what is slow, what will not fit and how sure we are, with a plain-language glossary.",
  path: "/run-it",
});

export default function RunItPage() {
  return (
    <>
      <PageIntro title="Can you run it?">
        Pick your machine. We say what works well, what is slow and what will not fit, and how sure we are.
      </PageIntro>
      <RigCheck asPage />
      <NextPage from="/run-it" />
    </>
  );
}
