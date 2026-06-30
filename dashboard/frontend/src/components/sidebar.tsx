"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { useStatus, usePositions } from "@/lib/api";
import { Home, FlaskConical, LayoutDashboard, Layers, Box, Menu, X } from "lucide-react";

const navItems = [
  { href: "/", label: "Home", icon: Home, exact: true },
  { href: "/lab", label: "Lab", icon: FlaskConical },
  { href: "/desk", label: "Desk", icon: LayoutDashboard },
  { href: "/experiments", label: "Experiments", icon: Layers },
  { href: "/architecture", label: "Architecture", icon: Box },
];

function HealthDot() {
  const { data } = useStatus();

  if (!data || data.status === "not_configured") {
    return <span className="inline-block h-2.5 w-2.5 rounded-full bg-neutral-500" />;
  }

  const lastRun = data.last_run_at;
  const lastStatus = data.last_run_status;

  let color = "bg-neutral-500";
  if (lastStatus === "error") {
    color = "bg-red-500";
  } else if (lastRun) {
    const elapsed = Date.now() - new Date(lastRun).getTime();
    const hours26 = 26 * 60 * 60 * 1000;
    color = elapsed < hours26 ? "bg-emerald-500" : "bg-yellow-500";
  }

  return <span className={cn("inline-block h-2.5 w-2.5 rounded-full", color)} />;
}

function PaperCapital() {
  const { data } = usePositions();

  if (!data || data.status === "not_started") {
    return null;
  }

  const totalValue = data.total_value || 0;
  const returnPct = ((totalValue / 10000) - 1) * 100;

  return (
    <div className="mt-4 border-t border-border pt-4 px-4">
      <p className="text-xs text-muted-foreground">Paper Capital</p>
      <p className="text-lg font-mono font-semibold">
        €{totalValue.toLocaleString(undefined, { maximumFractionDigits: 0 })}
      </p>
      <p className={cn(
        "text-xs font-mono",
        returnPct >= 0 ? "text-emerald-400" : "text-red-400"
      )}>
        {returnPct >= 0 ? "+" : ""}{returnPct.toFixed(2)}%
      </p>
    </div>
  );
}

function SidebarContent({ onNavigate }: { onNavigate?: () => void }) {
  const pathname = usePathname();

  return (
    <>
      <div className="flex items-center gap-2 px-4 py-5 border-b border-border">
        <div className="flex h-8 w-8 items-center justify-center rounded bg-accent">
          <span className="text-sm font-bold text-white">M</span>
        </div>
        <div>
          <p className="text-sm font-semibold">MAS Trading</p>
          <div className="flex items-center gap-1.5">
            <HealthDot />
            <span className="text-xs text-muted-foreground">Pipeline</span>
          </div>
        </div>
      </div>

      <nav className="flex-1 px-3 py-4 space-y-1">
        {navItems.map((item) => {
          const isActive = item.exact
            ? pathname === item.href
            : pathname === item.href || pathname.startsWith(item.href + "/");
          const Icon = item.icon;
          return (
            <Link
              key={item.href}
              href={item.href}
              onClick={onNavigate}
              className={cn(
                "flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors",
                isActive
                  ? "bg-accent/20 text-accent-foreground font-medium"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground"
              )}
            >
              <Icon className="h-4 w-4" />
              {item.label}
            </Link>
          );
        })}
      </nav>

      <PaperCapital />

      <div className="px-4 py-3 border-t border-border">
        <p className="text-[10px] text-muted-foreground">
          Multi-Agent System v1.0
        </p>
      </div>
    </>
  );
}

export function Sidebar() {
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <>
      {/* Mobile toggle */}
      <button
        className="fixed top-4 left-4 z-50 md:hidden flex h-9 w-9 items-center justify-center rounded-md border border-border bg-card"
        onClick={() => setMobileOpen(!mobileOpen)}
        aria-label="Toggle navigation"
      >
        {mobileOpen ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
      </button>

      {/* Mobile overlay */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 md:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Mobile sidebar */}
      <aside
        className={cn(
          "fixed left-0 top-0 z-40 flex h-screen w-56 flex-col border-r border-border bg-card transition-transform md:hidden",
          mobileOpen ? "translate-x-0" : "-translate-x-full"
        )}
      >
        <SidebarContent onNavigate={() => setMobileOpen(false)} />
      </aside>

      {/* Desktop sidebar */}
      <aside className="hidden md:flex fixed left-0 top-0 z-40 h-screen w-56 flex-col border-r border-border bg-card">
        <SidebarContent />
      </aside>
    </>
  );
}
