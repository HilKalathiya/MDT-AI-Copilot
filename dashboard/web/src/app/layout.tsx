import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "MDT AI Copilot — 5G Coverage Analytics",
  description:
    "Agentic AI analytics and RAG copilot for 5G MDT (Minimization of Drive Tests) — coverage anomaly detection, RSRP forecasting, and natural-language network assistant.",
  keywords: ["5G", "MDT", "AI", "RAG", "LangGraph", "OpenAirInterface", "RSRP", "coverage"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={inter.className}>{children}</body>
    </html>
  );
}
