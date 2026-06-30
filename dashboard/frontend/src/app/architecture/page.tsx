import { Card, CardContent } from "@/components/ui/card";
import {
  AlertTriangle,
  ArrowRight,
  Database,
  Server,
  Monitor,
} from "lucide-react";

const STACK = [
  {
    layer: "Agents",
    tech: "Python · XGBoost · FinBERT · Isolation Forest · HMM",
    reason: "Each agent runs in isolation with calibrated outputs — no shared state, no leakage between models",
  },
  {
    layer: "Judge (Meta-model)",
    tech: "Logistic Regression / XGBoost meta-model",
    reason: "Combines agent votes weighted by recent reliability — simple v1 before RL in v2",
  },
  {
    layer: "Risk Manager",
    tech: "Fractional Kelly (Half-Kelly, ρ=0.5) + ATR stop-loss",
    reason: "Position sizing based on edge magnitude, not binary signals — caps at 15% per ticker",
  },
  {
    layer: "Backtester",
    tech: "Custom walk-forward engine (Python)",
    reason: "Anti-leakage by design — train/val windows shift forward, no future data access possible",
  },
  {
    layer: "API",
    tech: "FastAPI · Pydantic v2 · slowapi",
    reason: "Typed schemas as single source of truth — OpenAPI spec auto-generates TypeScript types",
  },
  {
    layer: "Dashboard",
    tech: "Next.js 16 · React 19 · TanStack Query · Recharts · Tailwind 4",
    reason: "SSR-capable, stale-while-revalidate caching, responsive data visualization",
  },
  {
    layer: "Data",
    tech: "yfinance · Alpaca News API · SEC EDGAR",
    reason: "Free APIs with known limitations (T-1 lag) — documented, not hidden",
  },
];

const LIMITATIONS = [
  {
    title: "T-1 Data Lag",
    description:
      "All data comes from free APIs. Signals are computed after market close using end-of-day prices. Decisions apply to the next trading day's open.",
  },
  {
    title: "Survivorship Bias (documented)",
    description:
      "The stock universe is frozen as of January 1, 2018 to avoid selecting companies we know survived. This is a known limitation acknowledged by including 8 underperforming stocks (INTC, GILD, BIIB, WBA, BIDU, PYPL, ILMN).",
  },
  {
    title: "Paper Trading Only",
    description:
      "No real money is at risk. Trades are simulated with 0.08% commission per leg (approximating IBKR/Degiro). No slippage model is applied beyond commissions.",
  },
  {
    title: "Daily Signals Only",
    description:
      "The system generates one signal per ticker per day at market close. No intraday or multi-timeframe analysis. This is by design to avoid mixing incompatible signal horizons.",
  },
];

function PipelineNode({
  label,
  sub,
  Icon,
}: {
  label: string;
  sub: string;
  Icon: React.ComponentType<{ className?: string }>;
}) {
  return (
    <div className="flex flex-col items-center gap-1.5 text-center">
      <div className="flex h-12 w-12 items-center justify-center rounded-lg border border-border bg-card">
        <Icon className="h-5 w-5 text-accent-foreground" />
      </div>
      <p className="text-xs font-medium">{label}</p>
      <p className="text-[10px] text-muted-foreground max-w-[100px]">{sub}</p>
    </div>
  );
}

function PipelineArrow() {
  return <ArrowRight className="h-4 w-4 text-muted-foreground shrink-0 mt-2" />;
}

export default function ArchitecturePage() {
  return (
    <div className="space-y-12 max-w-5xl">
      <div>
        <h1 className="text-2xl font-bold">System Architecture</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Data flow, technology stack, and known limitations
        </p>
      </div>

      {/* Pipeline Diagram */}
      <section className="space-y-4">
        <h2 className="text-lg font-semibold">Data Flow</h2>
        <Card>
          <CardContent className="p-8">
            <div className="flex flex-wrap items-start justify-center gap-4">
              <PipelineNode
                label="Agents"
                sub="Isolated expert models"
                Icon={Database}
              />
              <PipelineArrow />
              <PipelineNode
                label="Judge"
                sub="Weighted ensemble"
                Icon={Server}
              />
              <PipelineArrow />
              <PipelineNode
                label="Risk Manager"
                sub="Kelly sizing + stops"
                Icon={AlertTriangle}
              />
              <PipelineArrow />
              <PipelineNode
                label="Backtester"
                sub="Walk-forward engine"
                Icon={Database}
              />
              <PipelineArrow />
              <PipelineNode
                label="Logs / JSON"
                sub="Experiments + trades"
                Icon={Database}
              />
              <PipelineArrow />
              <PipelineNode
                label="FastAPI"
                sub="REST + Pydantic"
                Icon={Server}
              />
              <PipelineArrow />
              <PipelineNode
                label="Dashboard"
                sub="Next.js + Recharts"
                Icon={Monitor}
              />
            </div>
            <p className="text-[10px] text-muted-foreground text-center mt-6">
              run_daily.py executes post-market close (cron) · Models are frozen after walk-forward selection · No .fit() in production
            </p>
          </CardContent>
        </Card>
      </section>

      {/* Tech Stack Table */}
      <section className="space-y-4">
        <h2 className="text-lg font-semibold">Technology Stack</h2>
        <div className="rounded-lg border border-border overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-muted/50">
                <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground w-[140px]">
                  Layer
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground w-[280px]">
                  Technology
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground">
                  Justification
                </th>
              </tr>
            </thead>
            <tbody>
              {STACK.map((row) => (
                <tr key={row.layer} className="border-b border-border/50">
                  <td className="px-4 py-3 font-medium text-sm">{row.layer}</td>
                  <td className="px-4 py-3 font-mono text-xs text-accent-foreground">
                    {row.tech}
                  </td>
                  <td className="px-4 py-3 text-xs text-muted-foreground leading-relaxed">
                    {row.reason}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Known Limitations */}
      <section className="space-y-4">
        <h2 className="text-lg font-semibold">Known Limitations</h2>
        <p className="text-sm text-muted-foreground">
          Honest documentation of constraints demonstrates engineering maturity.
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {LIMITATIONS.map((lim) => (
            <Card key={lim.title}>
              <CardContent className="p-5">
                <div className="flex items-start gap-3">
                  <AlertTriangle className="h-4 w-4 text-yellow-500 shrink-0 mt-0.5" />
                  <div>
                    <p className="text-sm font-semibold">{lim.title}</p>
                    <p className="text-xs text-muted-foreground leading-relaxed mt-1">
                      {lim.description}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>
    </div>
  );
}
