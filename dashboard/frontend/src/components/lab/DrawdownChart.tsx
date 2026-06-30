"use client";

import { useMemo } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { useEquityCurve } from "@/lib/api";

interface DrawdownPoint {
  date: string;
  drawdown: number;
}

function CustomTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const point = payload[0]?.payload as DrawdownPoint;
  return (
    <div className="rounded-md border border-border bg-card p-2 shadow-lg text-xs">
      <p className="text-foreground">{point.date}</p>
      <p className="text-red-400">Drawdown: {point.drawdown.toFixed(2)}%</p>
    </div>
  );
}

export function DrawdownChart() {
  const { data, isLoading } = useEquityCurve();

  const drawdownData = useMemo<DrawdownPoint[]>(() => {
    if (!data || data.length === 0) return [];

    let peak = data[0].portfolio_value;
    return data.map((point) => {
      if (point.portfolio_value > peak) peak = point.portfolio_value;
      const dd = ((point.portfolio_value - peak) / peak) * 100;
      return { date: point.date, drawdown: dd };
    });
  }, [data]);

  if (isLoading) {
    return (
      <div className="h-[200px] flex items-center justify-center text-muted-foreground">
        Loading drawdown...
      </div>
    );
  }

  if (drawdownData.length === 0) {
    return (
      <div className="rounded-lg border border-border bg-card p-8 text-center">
        <p className="text-muted-foreground text-sm">
          No drawdown data available · Run a backtest to generate equity curve data
        </p>
      </div>
    );
  }

  const minDD = Math.min(...drawdownData.map((d) => d.drawdown));

  return (
    <div className="w-full">
      <h2 className="text-lg font-semibold mb-4">Drawdown Underwater Plot</h2>
      <ResponsiveContainer width="100%" height={200}>
        <AreaChart data={drawdownData} margin={{ top: 5, right: 30, left: 10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2e2e2e" />
          <XAxis
            dataKey="date"
            tick={{ fill: "#a1a1a1", fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: "#2e2e2e" }}
            interval="preserveStartEnd"
            tickFormatter={(v: string) => v.substring(0, 7)}
          />
          <YAxis
            domain={[Math.floor(minDD - 2), 0]}
            tick={{ fill: "#a1a1a1", fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: "#2e2e2e" }}
            tickFormatter={(v: number) => `${v.toFixed(0)}%`}
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine y={0} stroke="#4b5563" />
          <Area
            type="monotone"
            dataKey="drawdown"
            stroke="#dc2626"
            fill="#dc2626"
            fillOpacity={0.4}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
