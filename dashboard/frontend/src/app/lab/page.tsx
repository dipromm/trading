"use client";

import Link from "next/link";
import { EquityChart } from "@/components/lab/EquityChart";
import { DrawdownChart } from "@/components/lab/DrawdownChart";
import { MetricsCards } from "@/components/lab/MetricsCards";
import { WalkForwardTable } from "@/components/lab/WalkForwardTable";
import { ReliabilityDiagram } from "@/components/lab/ReliabilityDiagram";
import { SHAPBeeswarm } from "@/components/lab/SHAPBeeswarm";
import { ErrorBoundary } from "@/components/ui/error-boundary";
import { useExperiments } from "@/lib/api";

function ExperimentsLink() {
  const { data } = useExperiments();
  const count = data?.length || 0;

  return (
    <Link
      href="/experiments"
      className="inline-flex items-center gap-2 text-sm text-accent-foreground hover:underline"
    >
      View all {count} experiments →
    </Link>
  );
}

export default function LabPage() {
  return (
    <div className="space-y-10 max-w-7xl">
      <div>
        <h1 className="text-2xl font-bold">The Quant Lab</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Walk-forward validation results and model explainability
        </p>
      </div>

      <ErrorBoundary fallbackMessage="Equity chart unavailable">
        <EquityChart />
      </ErrorBoundary>
      <ErrorBoundary fallbackMessage="Drawdown chart unavailable">
        <DrawdownChart />
      </ErrorBoundary>
      <ErrorBoundary fallbackMessage="Metrics unavailable">
        <MetricsCards />
      </ErrorBoundary>
      <ErrorBoundary fallbackMessage="Walk-forward table unavailable">
        <WalkForwardTable />
      </ErrorBoundary>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <ErrorBoundary fallbackMessage="Reliability diagram unavailable">
          <ReliabilityDiagram />
        </ErrorBoundary>
        <ErrorBoundary fallbackMessage="SHAP chart unavailable">
          <SHAPBeeswarm />
        </ErrorBoundary>
      </div>

      <div className="pt-4 border-t border-border">
        <ExperimentsLink />
      </div>
    </div>
  );
}
