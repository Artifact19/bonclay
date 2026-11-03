import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The dev-mode route indicator sits over the sidebar avatar and shows up in live demos.
  devIndicators: false,
  // The optimizer serves AVIF to browsers that accept it and WebP otherwise (Next's default is WebP only).
  images: { formats: ["image/avif", "image/webp"] },
  experimental: {
    optimizePackageImports: [
      "lucide-react",
      "recharts",
      "motion",
      "gsap",
      "clsx",
      "tailwind-merge",
      "swr",
      "sonner",
    ],
  },
};

export default nextConfig;
