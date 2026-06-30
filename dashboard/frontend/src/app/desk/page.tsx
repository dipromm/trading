"use client";

import { DataNotice } from "@/components/desk/DataNotice";
import { CouncilVerdictTable } from "@/components/desk/CouncilVerdictTable";
import { RiskManagement } from "@/components/desk/RiskManagement";
import { TradesTable } from "@/components/desk/TradesTable";
import { ErrorBoundary } from "@/components/ui/error-boundary";

export default function DeskPage() {
  return (
    <div className="space-y-8 max-w-7xl">
      <div>
        <h1 className="text-2xl font-bold">The Trading Desk</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Live model decisions and paper trading activity
        </p>
      </div>

      <ErrorBoundary fallbackMessage="Status data unavailable">
        <DataNotice />
      </ErrorBoundary>
      <ErrorBoundary fallbackMessage="Council verdict unavailable">
        <CouncilVerdictTable />
      </ErrorBoundary>
      <ErrorBoundary fallbackMessage="Risk management unavailable">
        <RiskManagement />
      </ErrorBoundary>
      <ErrorBoundary fallbackMessage="Trades log unavailable">
        <TradesTable />
      </ErrorBoundary>
    </div>
  );
}
