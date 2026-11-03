import { Toaster } from "sonner";
import { AuthProvider } from "@/components/AuthProvider";
import { LayoutWrapper } from "@/components/LayoutWrapper";
import { APP_FONT_CLASSES } from "@/lib/appFonts";
import { manrope } from "@/lib/brandFont";

/**
 * Shell of the app: the dashboard (sidebar, command palette), the studio and voices pages, the auth pages, auth state
 * and toasts. The public site and /verify do not render this layout, so none of this code ships to them. The app fonts
 * are declared on this wrapper (lib/appFonts.ts for Geist, lib/brandFont.ts for the Manrope wordmark).
 */
export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className={`${APP_FONT_CLASSES} ${manrope.variable}`}>
      <AuthProvider>
        <LayoutWrapper>{children}</LayoutWrapper>
      </AuthProvider>
      <Toaster position="top-right" richColors closeButton theme="system" />
    </div>
  );
}
