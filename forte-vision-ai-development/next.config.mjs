/** @type {import('next').NextConfig} */
const nextConfig = {
  // static export: served by the FastAPI backend from ./out — no Node server, no telemetry
  output: 'export',
  trailingSlash: true,
  images: {
    unoptimized: true,
  },
}

export default nextConfig
