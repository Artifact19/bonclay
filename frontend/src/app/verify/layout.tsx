import type { Metadata } from "next";
import { PublicMain } from "@/components/PublicMain";
import { manrope } from "@/lib/brandFont";

export const metadata: Metadata = {
  title: "Verify a clip - BonClay",
  description: "Upload an audio clip to check whether it was made with BonClay: an inaudible watermark and a signed receipt travel with every clip.",
};

/** Public and lean: no sign-in, no toaster. page-in is the route entrance the root template used to give it. */
export default function VerifyLayout({ children }: { children: React.ReactNode }) {
  return (
    <PublicMain>
      <div className={`page-in ${manrope.variable}`}>{children}</div>
    </PublicMain>
  );
}
