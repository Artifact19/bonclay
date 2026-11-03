import type { Metadata } from "next";
import { ErrorBoundary } from "@/components/ui";

// Home. Every page below sets its own title in its own layout.tsx, which overrides this one.
export const metadata: Metadata = { title: "Home · BonClay" };

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  return <ErrorBoundary>{children}</ErrorBoundary>;
}
