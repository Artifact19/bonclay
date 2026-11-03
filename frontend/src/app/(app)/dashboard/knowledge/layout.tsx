import type { Metadata } from "next";

export const metadata: Metadata = { title: "Documents · BonClay" };

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
