/** @type {import('next').NextConfig} */
const nextConfig = {
  // Standalone output for Docker production builds — produces a self-contained
  // server.js that doesn't need node_modules at runtime.
  output: 'standalone',
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 's4.anilist.co',
        pathname: '/**',
      },
      {
        protocol: 'https',
        hostname: 'img.anili.st',
        pathname: '/**',
      },
    ],
  },
};

export default nextConfig;
