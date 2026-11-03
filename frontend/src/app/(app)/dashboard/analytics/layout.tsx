import type { Metadata } from "next";

export const metadata: Metadata = { title: "Results · BonClay" };

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
