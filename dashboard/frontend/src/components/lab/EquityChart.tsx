"use client";

import {
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  ReferenceArea,
  Brush,
  Legend,
} from "recharts";
import { useEquityCurve, type EquityCurvePoint } from "@/lib/api";

const HISTORIC_EVENTS = [
  { date: "2020-03-23", label: "COVID Crash" },
  { date: "2022-03-16", label: "Rate Hike Cycle" },
  { date: "2023-03-10", label: "SVB Crisis" },
];

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;

  const point = payload[0]?.payload as EquityCurvePoint | undefined;
  if (!point) return null;

  return (
    <div className="rounded-md border border-border bg-card p-3 shadow-lg text-xs">
      <p className="font-medium text-foreground mb-1">{point.date}</p>
      <p className="text-blue-400">
        MAS: {point.portfolio_value?.toFixed(1)} ({((point.portfolio_value / 100 - 1) * 100).toFixed(1)}%)
      </p>
      {point.buyhold_value != null && (
        <p className="text-neutral-400">
          B&H: {point.buyhold_value.toFixed(1)} ({((point.buyhold_value / 100 - 1) * 100).toFixed(1)}%)
        </p>
      )}
      {point.sma_value != null && (
        <p className="text-amber-400">
          SMA: {point.sma_value.toFixed(1)} ({((point.sma_value / 100 - 1) * 100).toFixed(1)}%)
        </p>
      )}
      <p className="text-muted-foreground mt-1 capitalize">{point.period.replace("_", " ")}</p>
    </div>
  );
}

export function EquityChart() {
  const { data, isLoading, error } = useEquityCurve();

  if (isLoading) {
    return (
      <div className="h-[400px] flex items-center justify-center text-muted-foreground">
        Loading equity curve...
      </div>
    );
  }

  if (error || !data || data.length === 0) {
    return (
      <div className="h-[400px] flex items-center justify-center text-muted-foreground">
        No equity curve data available
      </div>
    );
  }

  const wfEnd = data.find((d) => d.period === "holdout")?.date || "2025-01-01";
  const holdoutEnd = data.find((d) => d.period === "paper_trading")?.date;

  const minDate = data[0]?.date;
  const maxDate = data[data.length - 1]?.date;

  return (
    <div className="w-full">
      <h2 className="text-lg font-semibold mb-4">Equity Curve vs Baselines</h2>
      <ResponsiveContainer width="100%" height={420}>
        <ComposedChart data={data} margin={{ top: 10, right: 30, left: 10, bottom: 30 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2e2e2e" />

          {/* Period background areas */}
          <ReferenceArea
            x1={minDate}
            x2={wfEnd}
            fill="rgba(255,255,255,0.03)"
            fillOpacity={1}
          />
          {holdoutEnd && (
            <ReferenceArea
              x1={wfEnd}
              x2={holdoutEnd}
              fill="rgba(0,0,0,0.3)"
              fillOpacity={1}
            />
          )}

          <XAxis
            dataKey="date"
            tick={{ fill: "#a1a1a1", fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: "#2e2e2e" }}
            interval="preserveStartEnd"
            tickFormatter={(v: string) => v.substring(0, 7)}
          />
          <YAxis
            scale="log"
            domain={["auto", "auto"]}
            tick={{ fill: "#a1a1a1", fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: "#2e2e2e" }}
            tickFormatter={(v: number) => v.toFixed(0)}
          />

          <Tooltip content={<CustomTooltip />} />
          <Legend
            verticalAlign="top"
            height={36}
            wrapperStyle={{ fontSize: 12, color: "#a1a1a1" }}
          />

          {/* Period separator */}
          <ReferenceLine
            x={wfEnd}
            stroke="#4b5563"
            strokeDasharray="4 4"
            label={{ value: "Holdout →", position: "top", fill: "#6b7280", fontSize: 10 }}
          />

          {/* Historic events */}
          {HISTORIC_EVENTS.map((evt) => (
            <ReferenceLine
              key={evt.date}
              x={evt.date}
              stroke="#374151"
              strokeDasharray="2 2"
              label={{ value: evt.label, position: "insideTopRight", fill: "#6b7280", fontSize: 9, angle: -90 }}
            />
          ))}

          {/* Strategy lines */}
          <Line
            type="monotone"
            dataKey="portfolio_value"
            name="MAS"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={false}
            connectNulls
          />
          <Line
            type="monotone"
            dataKey="buyhold_value"
            name="Buy & Hold"
            stroke="#6b7280"
            strokeWidth={1.5}
            dot={false}
            strokeDasharray="4 2"
            connectNulls
          />
          <Line
            type="monotone"
            dataKey="sma_value"
            name="SMA Crossover"
            stroke="#f59e0b"
            strokeWidth={1.5}
            dot={false}
            strokeDasharray="2 2"
            connectNulls
          />

          <Brush
            dataKey="date"
            height={30}
            stroke="#3b82f6"
            fill="#1a1a1a"
            tickFormatter={(v: string) => v.substring(0, 7)}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
