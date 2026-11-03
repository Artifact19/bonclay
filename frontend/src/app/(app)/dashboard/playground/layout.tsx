import type { Metadata } from "next";

export const metadata: Metadata = { title: "Talk to your agent · BonClay" };

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
