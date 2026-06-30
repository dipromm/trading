import type { Metadata } from "next";
import "./globals.css";
import { QueryProvider } from "@/lib/query-provider";
import { Sidebar } from "@/components/sidebar";

export const metadata: Metadata = {
  title: "MAS Trading System",
  description: "Multi-Agent Trading System — Walk-Forward, Holdout & Paper Trading Dashboard",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased">
        <QueryProvider>
          <div className="flex min-h-screen">
            <Sidebar />
            <main className="md:ml-56 flex-1 p-4 pt-16 md:p-8">
              {children}
            </main>
          </div>
        </QueryProvider>
      </body>
    </html>
  );
}
