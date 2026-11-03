import type { Metadata } from "next";

export const metadata: Metadata = { title: "Try a document · BonClay" };

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
