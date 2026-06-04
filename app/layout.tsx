import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "BeZeus",
  description: "A camera-controlled thunder interface for summoning lightning by hand.",
  applicationName: "BeZeus",
  keywords: [
    "camera interface",
    "gesture control",
    "hand tracking",
    "WebGL lightning",
    "MediaPipe prototype",
  ],
  alternates: {
    canonical: "/",
  },
  openGraph: {
    title: "BeZeus",
    description: "Camera, hand tracking, WebGL fallback, and gesture-based lightning prototype.",
    type: "website",
  },
  robots: {
    index: true,
    follow: true,
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
