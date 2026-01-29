import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  turbopack: {
    // Ensure the correct project root to avoid stale caches when multiple lockfiles exist
    root: __dirname,
  },
  // Include puppeteer packages as external for PDF generation
  serverExternalPackages: ['puppeteer-core', '@sparticuz/chromium-min'],
};

export default nextConfig;
