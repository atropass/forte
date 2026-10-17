/** @type {import('next').NextConfig} */
const nextConfig = {
  // static export: served by the FastAPI backend (or nginx) from ./out — no Node server, no telemetry
  output: 'export',
  trailingSlash: true,
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
}

export default nextConfig
