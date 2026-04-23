import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'AniMind — Anime RAG Chatbot',
  description:
    'An AI-powered anime knowledge chatbot built with LangGraph, Qdrant, and Qwen3-Reranker. Ask anything about anime!',
  openGraph: {
    title: 'AniMind — Anime RAG Chatbot',
    description: 'AI-powered anime knowledge chatbot',
    type: 'website',
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        {/* Wallpaper background */}
        <div
          className="fixed inset-0 bg-cover bg-center bg-no-repeat"
          style={{ backgroundImage: "url('/kaguya-wpp.jpg')" }}
          aria-hidden="true"
        />
        {/* Dim overlay */}
        <div className="fixed inset-0 bg-black/35" aria-hidden="true" />
        {/* Content */}
        <div className="relative z-10 h-screen">
          {children}
        </div>
      </body>
    </html>
  );
}
