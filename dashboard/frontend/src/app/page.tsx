"use client";

import Link from "next/link";
import { useMetrics, type PeriodMetrics } from "@/lib/api";
import { Card, CardContent } from "@/components/ui/card";
import {
  TrendingUp,
  Newspaper,
  Shield,
  AlertTriangle,
  FlaskConical,
  LayoutDashboard,
} from "lucide-react";

const AGENTS = [
  {
    name: "El Matemático",
    role: "Technical analysis — price and volume trends",
    tech: "XGBoost + calibrated probabilities",
    Icon: TrendingUp,
  },
  {
    name: "El Analista",
    role: "Sentiment analysis from financial headlines",
    tech: "FinBERT (HuggingFace)",
    Icon: Newspaper,
  },
  {
    name: "El Cazador",
    role: "Insider trading detection via SEC filings",
    tech: "SEC EDGAR Form 4 parser",
    Icon: Shield,
  },
  {
    name: "El Conspiranoico",
    role: "Anomalous market regime detection and veto",
    tech: "Isolation Forest + HMM",
    Icon: AlertTriangle,
  },
];

function fmt(v: number | null | undefined, suffix = ""): string {
  if (v == null) return "—";
  return `${v >= 0 && suffix !== "%" ? "" : ""}${v.toFixed(2)}${suffix}`;
}

function fmtPct(v: number | null | undefined): string {
  if (v == null) return "—";
  return `${v.toFixed(1)}%`;
}

export default function HomePage() {
  const { data: holdoutData, isLoading } = useMetrics("holdout");

  const holdout = (holdoutData as PeriodMetrics)?.mas;
  const bh = (holdoutData as PeriodMetrics)?.buy_and_hold;

  const outperformance =
    holdout?.annualized_return_pct != null && bh?.annualized_return_pct != null
      ? holdout.annualized_return_pct - bh.annualized_return_pct
      : null;

  const metrics = [
    { label: "Sharpe Ratio", value: fmt(holdout?.sharpe_ratio), sub: "Holdout period" },
    { label: "Max Drawdown", value: fmtPct(holdout?.max_drawdown_pct), sub: "Worst peak-to-trough" },
    { label: "Calmar Ratio", value: fmt(holdout?.calmar_ratio), sub: "Return / Max DD" },
    {
      label: "vs Buy & Hold",
      value: outperformance != null ? `${outperformance >= 0 ? "+" : ""}${outperformance.toFixed(1)}pp` : "—",
      sub: "Annualized outperformance",
    },
  ];

  return (
    <div className="max-w-5xl space-y-16 py-8">
      {/* Hero */}
      <section className="space-y-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-accent">
            <span className="text-lg font-bold text-white">M</span>
          </div>
          <div>
            <h1 className="text-3xl font-bold tracking-tight">MAS Trading System</h1>
            <span className="text-xs text-muted-foreground font-mono">v1.0 · Multi-Agent System</span>
          </div>
        </div>
        <p className="text-muted-foreground max-w-2xl leading-relaxed">
          An AI-powered ensemble of specialized agents that analyze U.S. equities from
          isolated perspectives — technical patterns, news sentiment, insider activity,
          and market regime — then a central Judge combines their votes into daily
          trading decisions validated through rigorous walk-forward backtesting.
        </p>
      </section>

      {/* Holdout Metrics */}
      <section className="space-y-4">
        <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
          Holdout Period Results (Blind Test)
        </h2>
        {isLoading ? (
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {[...Array(4)].map((_, i) => (
              <Card key={i}>
                <CardContent className="p-6">
                  <div className="h-4 w-20 bg-muted rounded animate-pulse mb-3" />
                  <div className="h-8 w-16 bg-muted rounded animate-pulse" />
                </CardContent>
              </Card>
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {metrics.map((m) => (
              <Card key={m.label}>
                <CardContent className="p-6">
                  <p className="text-xs text-muted-foreground mb-1">{m.label}</p>
                  <p className="text-3xl font-mono font-bold">{m.value}</p>
                  <p className="text-[10px] text-muted-foreground mt-1">{m.sub}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </section>

      {/* Agent Cards */}
      <section className="space-y-4">
        <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
          The Council of Agents
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {AGENTS.map((agent) => (
            <Card key={agent.name}>
              <CardContent className="p-5 flex items-start gap-4">
                <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-muted">
                  <agent.Icon className="h-4 w-4 text-accent-foreground" />
                </div>
                <div className="min-w-0">
                  <p className="text-sm font-semibold">{agent.name}</p>
                  <p className="text-xs text-muted-foreground leading-relaxed">{agent.role}</p>
                  <p className="text-[10px] font-mono text-muted-foreground mt-1">{agent.tech}</p>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      {/* CTAs */}
      <section className="flex flex-wrap gap-4">
        <Link
          href="/lab"
          className="inline-flex items-center gap-2 rounded-lg bg-accent px-6 py-3 text-sm font-medium text-white transition-colors hover:bg-accent/80"
        >
          <FlaskConical className="h-4 w-4" />
          View the Lab
        </Link>
        <Link
          href="/desk"
          className="inline-flex items-center gap-2 rounded-lg border border-border bg-card px-6 py-3 text-sm font-medium text-foreground transition-colors hover:bg-muted"
        >
          <LayoutDashboard className="h-4 w-4" />
          View the Desk
        </Link>
      </section>
    </div>
  );
}
