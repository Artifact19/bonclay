import type { Metadata } from "next";

export const metadata: Metadata = { title: "Agents · BonClay" };

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
