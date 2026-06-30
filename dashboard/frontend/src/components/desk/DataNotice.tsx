"use client";

import { useStatus, useMarketClose } from "@/lib/api";
import { cn } from "@/lib/utils";

export function DataNotice() {
  const { data: status } = useStatus();
  const { data: marketClose } = useMarketClose();

  const lastRunStatus = status?.last_run_status;
  const lastRunAt = status?.last_run_at;

  let healthColor = "bg-neutral-500";
  let healthLabel = "Unknown";

  if (status?.status === "not_configured") {
    healthColor = "bg-neutral-500";
    healthLabel = "Not configured";
  } else if (lastRunStatus === "error") {
    healthColor = "bg-red-500";
    healthLabel = "Error";
  } else if (lastRunAt) {
    const elapsed = Date.now() - new Date(lastRunAt).getTime();
    const hours26 = 26 * 60 * 60 * 1000;
    if (elapsed < hours26) {
      healthColor = "bg-emerald-500";
      healthLabel = "Healthy";
    } else {
      healthColor = "bg-yellow-500";
      healthLabel = "Stale";
    }
  }

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className={cn("h-3 w-3 rounded-full animate-pulse", healthColor)} />
          <div>
            <p className="text-sm font-medium">{healthLabel}</p>
            <p className="text-xs text-muted-foreground">
              {marketClose ? (
                <>
                  Last market close: <span className="font-medium text-foreground">{marketClose.last_market_close}</span>
                  {marketClose.pipeline_ran_at && (
                    <> · Pipeline ran: {marketClose.pipeline_ran_at}</>
                  )}
                </>
              ) : (
                "Loading market data..."
              )}
            </p>
          </div>
        </div>

        {lastRunStatus === "error" && (
          <button className="text-xs text-red-400 border border-red-900 rounded px-2 py-1 hover:bg-red-950 transition-colors">
            View error log
          </button>
        )}
      </div>
    </div>
  );
}
