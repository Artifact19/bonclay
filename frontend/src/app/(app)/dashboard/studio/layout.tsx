import type { Metadata } from "next";

export const metadata: Metadata = { title: "Voice Studio · BonClay" };

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
