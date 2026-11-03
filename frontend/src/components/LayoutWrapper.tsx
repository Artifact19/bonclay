"use client";

import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";
import { BonClaySidebar } from "@/components/BonClaySidebar";
import { Search } from "lucide-react";
import { ErrorBoundary } from "@/components/ui";
import { CommandPalette } from "@/components/CommandPalette";
import { SkipLink } from "@/components/SkipLink";
import { TourGuide, TourTriggerButton } from "@/components/TourGuide";

const COLLAPSE_KEY = "bonclay-sidebar-collapsed";

export function LayoutWrapper({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const showSidebar = pathname.startsWith("/dashboard");
  const [collapsed, setCollapsed] = useState(false);

  useEffect(() => {
    try {
      // Hydration-safe: read persisted preference after mount.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setCollapsed(localStorage.getItem(COLLAPSE_KEY) === "1");
    } catch {
      /* default expanded */
    }
  }, []);

  const [paletteOpen, setPaletteOpen] = useState(false);
  const [tourOpen, setTourOpen] = useState(false);
  const [isMac, setIsMac] = useState(false);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setIsMac(/Mac|iPhone|iPad/i.test(navigator.platform || navigator.userAgent));
  }, []);

  // Cmd/Ctrl+K opens the palette anywhere in the dashboard.
  useEffect(() => {
    if (!showSidebar) return;
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && !e.altKey && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen((o) => !o);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [showSidebar]);

  const toggle = () => {
    setCollapsed((c) => {
      try {
        localStorage.setItem(COLLAPSE_KEY, c ? "0" : "1");
      } catch {
        /* ignore */
      }
      return !c;
    });
  };

  return (
    <>
      <SkipLink />
      {showSidebar && <BonClaySidebar collapsed={collapsed} onToggleCollapsed={toggle} />}
      {showSidebar && (
        <>
          <button
            type="button"
            onClick={() => setPaletteOpen(true)}
            aria-label="Search and commands"
            aria-haspopup="dialog"
            className="fixed right-3 top-2 z-40 grid size-10 place-items-center rounded-lg text-text-primary hover:bg-bg-elevated md:hidden"
          >
            <Search className="size-5" aria-hidden />
          </button>
          <div className="fixed right-4 top-2.5 z-30 hidden items-center gap-2 md:flex">
            <TourTriggerButton onClick={() => setTourOpen(true)} label="Tour guide" />
            <button
              type="button"
              onClick={() => setPaletteOpen(true)}
              aria-label="Search and commands"
              aria-haspopup="dialog"
              className="flex h-8.5 items-center gap-2 rounded-full border-[1.5px] border-[#141414]/30 bg-bg-surface/90 px-3 text-xs text-text-secondary backdrop-blur transition-colors hover:border-[#141414] hover:text-text-primary"
            >
              <Search className="size-3.5" aria-hidden />
              <span>Search</span>
              <kbd className="font-sans text-[11px] opacity-70">{isMac ? "\u2318K" : "Ctrl K"}</kbd>
            </button>
          </div>
          <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} />
          <TourGuide mode="dashboard" open={tourOpen} onClose={() => setTourOpen(false)} />
        </>
      )}
      <main
        id="main-content"
        tabIndex={-1}
        className={`relative z-10 min-h-screen outline-none transition-[padding] duration-300 ease-[cubic-bezier(0.16,1,0.3,1)] ${
          showSidebar ? `pt-14 md:pt-0 ${collapsed ? "md:pl-16" : "md:pl-60"}` : ""
        }`}
      >
        <ErrorBoundary>{children}</ErrorBoundary>
      </main>
    </>
  );
}
