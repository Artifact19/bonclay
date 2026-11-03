import { Manrope } from "next/font/google";

/**
 * The brand wordmark face (components/Logo), Regular 400 only. It is the one font the public pages load from the shared
 * font modules. It lives apart from lib/appFonts.ts because importing that module also registers Geist and Geist Mono,
 * and next/font preloads each face it registers.
 */
export const manrope = Manrope({
  subsets: ["latin"],
  weight: "400",
  variable: "--font-manrope",
  display: "swap",
});
