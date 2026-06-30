"use client";

import { useMetrics, type PeriodMetrics, type MetricsData } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface MetricCardProps {
  label: string;
  wfValue: number | null | undefined;
  holdoutValue: number | null | undefined;
  bhValue?: number | null;
  format?: (v: number) => string;
  target?: { op: "gt" | "lt"; value: number };
}

function MetricCard({ label, wfValue, holdoutValue, bhValue, format, target }: MetricCardProps) {
  const fmt = format || ((v: number) => v.toFixed(2));

  function meetsTarget(val: number | null | undefined): boolean | null {
    if (val == null || !target) return null;
    return target.op === "gt" ? val > target.value : val < target.value;
  }

  const wfMet = meetsTarget(wfValue);
  const holdoutMet = meetsTarget(holdoutValue);

  return (
    <Card>
      <CardHeader>
        <CardTitle>{label}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-[10px] text-muted-foreground mb-1">Walk-Forward</p>
            <div className="flex items-center gap-1.5">
              {wfMet !== null && (
                <span className={cn("h-2 w-2 rounded-full", wfMet ? "bg-emerald-500" : "bg-red-500")} />
              )}
              <span className="text-xl font-mono font-semibold">
                {wfValue != null ? fmt(wfValue) : "—"}
              </span>
            </div>
          </div>
          <div>
            <p className="text-[10px] text-muted-foreground mb-1">Holdout Ciego</p>
            <div className="flex items-center gap-1.5">
              {holdoutMet !== null && (
                <span className={cn("h-2 w-2 rounded-full", holdoutMet ? "bg-emerald-500" : "bg-red-500")} />
              )}
              <span className="text-xl font-mono font-semibold">
                {holdoutValue != null ? fmt(holdoutValue) : "—"}
              </span>
            </div>
          </div>
        </div>
        {bhValue != null && (
          <p className="text-[10px] text-muted-foreground mt-2">
            Buy & Hold: <span className="font-mono">{fmt(bhValue)}</span>
          </p>
        )}
      </CardContent>
    </Card>
  );
}

export function MetricsCards() {
  const { data: wfData, isLoading: wfLoading } = useMetrics("walkforward");
  const { data: holdoutData, isLoading: holdoutLoading } = useMetrics("holdout");

  if (wfLoading || holdoutLoading) {
    return (
      <div className="w-full">
        <h2 className="text-lg font-semibold mb-4">Institutional Metrics</h2>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {[...Array(8)].map((_, i) => (
            <Card key={i}>
              <CardHeader><CardTitle><div className="h-3 w-20 bg-muted rounded animate-pulse" /></CardTitle></CardHeader>
              <CardContent>
                <div className="h-6 w-16 bg-muted rounded animate-pulse" />
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  const wf = (wfData as PeriodMetrics)?.mas || (wfData as MetricsData);
  const holdout = (holdoutData as PeriodMetrics)?.mas;
  const bhWf = (wfData as PeriodMetrics)?.buy_and_hold;

  return (
    <div className="w-full">
      <h2 className="text-lg font-semibold mb-4">Institutional Metrics</h2>
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          label="Sharpe Ratio"
          wfValue={wf?.sharpe_ratio}
          holdoutValue={holdout?.sharpe_ratio}
          bhValue={bhWf?.sharpe_ratio}
          format={(v) => v.toFixed(3)}
          target={{ op: "gt", value: 1.0 }}
        />
        <MetricCard
          label="Sortino Ratio"
          wfValue={wf?.sortino_ratio}
          holdoutValue={holdout?.sortino_ratio}
          format={(v) => v.toFixed(3)}
        />
        <MetricCard
          label="Calmar Ratio"
          wfValue={wf?.calmar_ratio}
          holdoutValue={holdout?.calmar_ratio}
          bhValue={bhWf?.calmar_ratio}
          format={(v) => v.toFixed(3)}
        />
        <MetricCard
          label="Max Drawdown"
          wfValue={wf?.max_drawdown_pct}
          holdoutValue={holdout?.max_drawdown_pct}
          bhValue={bhWf?.max_drawdown_pct}
          format={(v) => `${v.toFixed(1)}%`}
          target={{ op: "gt", value: -25 }}
        />
        <MetricCard
          label="Ann. Return"
          wfValue={wf?.annualized_return_pct}
          holdoutValue={holdout?.annualized_return_pct}
          bhValue={bhWf?.annualized_return_pct}
          format={(v) => `${v.toFixed(1)}%`}
        />
        <MetricCard
          label="Win Rate"
          wfValue={wf?.win_rate_pct}
          holdoutValue={holdout?.win_rate_pct}
          bhValue={bhWf?.win_rate_pct}
          format={(v) => `${v.toFixed(1)}%`}
        />
        <MetricCard
          label="Profit Factor"
          wfValue={wf?.profit_factor}
          holdoutValue={holdout?.profit_factor}
          format={(v) => v.toFixed(3)}
        />
        <MetricCard
          label="Total Return"
          wfValue={wf?.total_return_pct}
          holdoutValue={holdout?.total_return_pct}
          bhValue={bhWf?.total_return_pct}
          format={(v) => `${v.toFixed(1)}%`}
        />
      </div>
    </div>
  );
}
