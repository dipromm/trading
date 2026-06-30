"use client";

import { useMemo } from "react";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from "recharts";
import { usePositions, useCouncilVerdict, type PositionEntry, type PositionsData } from "@/lib/api";
import "katex/dist/katex.min.css";
import { InlineMath, BlockMath } from "react-katex";

const COLORS: Record<string, string> = {
  Equity: "#3b82f6",
  Bond: "#6366f1",
  Gold: "#eab308",
  Defensive: "#10b981",
  International: "#8b5cf6",
  Cash: "#4b5563",
};

const ASSET_CLASS_MAP: Record<string, string> = {
  TLT: "Bond", IEF: "Bond",
  GLD: "Gold",
  XLU: "Defensive", XLP: "Defensive",
  EFA: "International",
};

function getAssetClass(ticker: string): string {
  return ASSET_CLASS_MAP[ticker] || "Equity";
}

function positionValue(pos: {
  current_value?: number | null;
  allocated_eur?: number | null;
  quantity?: number;
  entry_price?: number;
}): number {
  if (pos.current_value != null && pos.current_value > 0) return pos.current_value;
  if (pos.allocated_eur != null && pos.allocated_eur > 0) return pos.allocated_eur;
  return (pos.quantity ?? 0) * (pos.entry_price ?? 0);
}

function toPositionList(
  positions: PositionsData["positions"] | Record<string, Omit<PositionEntry, "ticker"> & { allocated_eur?: number }>
): PositionEntry[] {
  if (Array.isArray(positions)) return positions;
  if (positions && typeof positions === "object") {
    return Object.entries(positions).map(([ticker, pos]) => ({
      ticker,
      quantity: pos.quantity ?? 0,
      entry_price: pos.entry_price ?? 0,
      entry_date: pos.entry_date ?? "",
      current_value: positionValue(pos),
      unrealized_pnl: pos.unrealized_pnl,
    }));
  }
  return [];
}

export function RiskManagement() {
  const { data: positions, isLoading: posLoading } = usePositions();
  const { data: verdict, isLoading: verdictLoading } = useCouncilVerdict();

  const donutData = useMemo(() => {
    if (!positions || positions.status === "not_started") {
      return [{ name: "Cash", value: 10000 }];
    }

    const list = toPositionList(
      positions.positions as PositionsData["positions"] | Record<string, Omit<PositionEntry, "ticker"> & { allocated_eur?: number }>
    );

    const groups: Record<string, number> = {};
    for (const pos of list) {
      const cls = getAssetClass(pos.ticker);
      groups[cls] = (groups[cls] || 0) + positionValue(pos);
    }

    const cash = Math.max(0, positions.cash || 0);
    if (cash > 0) {
      groups["Cash"] = cash;
    }

    const slices = Object.entries(groups)
      .map(([name, value]) => ({ name, value: Math.round(value) }))
      .filter((d) => d.value > 0);

    return slices.length > 0 ? slices : [{ name: "Cash", value: 10000 }];
  }, [positions]);

  const topKellyDecision = useMemo(() => {
    if (!verdict?.decisions?.length) return null;
    const sorted = [...verdict.decisions].sort((a, b) => b.kelly_fraction - a.kelly_fraction);
    return sorted[0];
  }, [verdict]);

  if (posLoading && verdictLoading) {
    return (
      <div className="w-full">
        <h2 className="text-lg font-semibold mb-4">Risk Management</h2>
        <div className="text-muted-foreground text-sm">Loading risk data...</div>
      </div>
    );
  }

  return (
    <div className="w-full">
      <h2 className="text-lg font-semibold mb-4">Risk Management</h2>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Donut chart */}
        <div className="rounded-lg border border-border bg-card p-4">
          <h3 className="text-sm font-medium text-muted-foreground mb-3">Capital Allocation</h3>
          <ResponsiveContainer width="100%" height={240}>
            <PieChart>
              <Pie
                data={donutData}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={90}
                dataKey="value"
                nameKey="name"
                paddingAngle={2}
              >
                {donutData.map((entry) => (
                  <Cell key={entry.name} fill={COLORS[entry.name] || "#6b7280"} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{ backgroundColor: "#1a1a1a", border: "1px solid #2e2e2e" }}
                formatter={(value: any) => [`€${Number(value).toLocaleString()}`, ""]}
              />
              <Legend
                verticalAlign="bottom"
                height={36}
                wrapperStyle={{ fontSize: 11, color: "#a1a1a1" }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Kelly formula */}
        <div className="rounded-lg border border-border bg-card p-4">
          <h3 className="text-sm font-medium text-muted-foreground mb-3">Fractional Kelly Criterion</h3>
          <div className="flex items-center justify-center py-4">
            <BlockMath math="f^* = \rho \times \left(p - \frac{1-p}{b}\right)" />
          </div>
          {topKellyDecision && (
            <div className="mt-4 rounded-md bg-muted/50 p-3">
              <p className="text-xs text-muted-foreground mb-2">
                Live example: <span className="font-mono font-medium text-foreground">{topKellyDecision.ticker}</span> (highest f* today)
              </p>
              <div className="flex items-center gap-2 text-sm">
                <InlineMath math={`\\rho=${topKellyDecision.kelly_inputs.rho}`} />
                <span className="text-muted-foreground">,</span>
                <InlineMath math={`p=${topKellyDecision.kelly_inputs.p.toFixed(2)}`} />
                <span className="text-muted-foreground">,</span>
                <InlineMath math={`b=${topKellyDecision.kelly_inputs.b.toFixed(2)}`} />
                <span className="text-muted-foreground">→</span>
                <InlineMath math={`f^*=${topKellyDecision.kelly_fraction.toFixed(4)}`} />
              </div>
            </div>
          )}
          {!topKellyDecision && (
            <p className="text-xs text-muted-foreground mt-4">
              Live Kelly values will appear when paper trading starts.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
