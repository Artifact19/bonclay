import { redirect } from "next/navigation";

/** The legacy studio now lives at /dashboard/studio. Query params (voice, mode) carry over. */
export default async function LegacyStudioRedirect({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const sp = await searchParams;
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(sp)) {
    if (typeof v === "string") qs.set(k, v);
  }
  const q = qs.toString();
  redirect(q ? `/dashboard/studio?${q}` : "/dashboard/studio");
}
