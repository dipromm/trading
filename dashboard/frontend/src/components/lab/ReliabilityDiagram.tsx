"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { useReliabilityDiagram } from "@/lib/api";

export function ReliabilityDiagram() {
  const { data, isLoading } = useReliabilityDiagram();

  if (isLoading) {
    return <div className="text-muted-foreground text-sm">Loading calibration...</div>;
  }

  if (!data || data.status === "not_generated") {
    return (
      <div className="text-muted-foreground text-sm">
        Reliability diagram not yet generated. Run{" "}
        <code className="font-mono">python scripts/generate_xai_artifacts.py</code> to compute.
      </div>
    );
  }

  if (!data.calibration_curve?.length) {
    return (
      <div className="rounded-lg border border-border bg-card/50 p-6 text-center space-y-1">
        <p className="text-sm text-muted-foreground">
          Calibration data accumulating from live paper trading.
        </p>
        <p className="text-xs text-muted-foreground/60">
          Reliability diagram will populate after several trading sessions.
        </p>
      </div>
    );
  }

  const perfectLine = Array.from({ length: 11 }, (_, i) => ({
    mean_predicted: i / 10,
    perfect: i / 10,
  }));

  const chartData = data.calibration_curve.map((pt) => ({
    mean_predicted: pt.mean_predicted,
    fraction_of_positives: pt.fraction_of_positives,
    perfect: pt.mean_predicted,
  }));

  return (
    <div>
      <h3 className="text-base font-semibold mb-3">Reliability Diagram (Calibration)</h3>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2e2e2e" />
          <XAxis
            dataKey="mean_predicted"
            tick={{ fill: "#a1a1a1", fontSize: 11 }}
            tickLine={false}
            label={{ value: "Mean Predicted Probability", position: "bottom", fill: "#6b7280", fontSize: 11 }}
          />
          <YAxis
            tick={{ fill: "#a1a1a1", fontSize: 11 }}
            tickLine={false}
            domain={[0, 1]}
            label={{ value: "Fraction Positive", angle: -90, position: "left", fill: "#6b7280", fontSize: 11 }}
          />
          <Tooltip
            contentStyle={{ backgroundColor: "#1a1a1a", border: "1px solid #2e2e2e" }}
            labelStyle={{ color: "#a1a1a1" }}
          />
          <Legend verticalAlign="top" height={30} wrapperStyle={{ fontSize: 11 }} />
          <Line
            type="monotone"
            dataKey="perfect"
            name="Perfect Calibration"
            stroke="#4b5563"
            strokeDasharray="4 4"
            dot={false}
          />
          <Line
            type="monotone"
            dataKey="fraction_of_positives"
            name="Model"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={{ fill: "#3b82f6", r: 3 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
