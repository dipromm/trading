"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  flexRender,
  createColumnHelper,
  type SortingState,
} from "@tanstack/react-table";
import { useExperiments, type ExperimentSummary } from "@/lib/api";
import { cn } from "@/lib/utils";

const columnHelper = createColumnHelper<ExperimentSummary>();

const columns = [
  columnHelper.accessor("experiment_id", {
    header: "Experiment",
    cell: (info) => (
      <span className="font-mono text-xs truncate max-w-[260px] inline-block">
        {info.getValue()}
      </span>
    ),
  }),
  columnHelper.accessor("sharpe_ratio", {
    header: "Sharpe",
    cell: (info) => {
      const v = info.getValue();
      return <span className="font-mono">{v != null ? v.toFixed(3) : "—"}</span>;
    },
  }),
  columnHelper.accessor("max_drawdown_pct", {
    header: "Max DD",
    cell: (info) => {
      const v = info.getValue();
      return (
        <span className="font-mono text-red-400">
          {v != null ? `${v.toFixed(1)}%` : "—"}
        </span>
      );
    },
  }),
  columnHelper.accessor("calmar_ratio", {
    header: "Calmar",
    cell: (info) => {
      const v = info.getValue();
      return <span className="font-mono">{v != null ? v.toFixed(3) : "—"}</span>;
    },
  }),
  columnHelper.accessor("total_return_pct", {
    header: "Total Return",
    cell: (info) => {
      const v = info.getValue();
      if (v == null) return <span className="text-muted-foreground">—</span>;
      return (
        <span className={cn("font-mono", v >= 0 ? "text-emerald-400" : "text-red-400")}>
          {v >= 0 ? "+" : ""}{v.toFixed(1)}%
        </span>
      );
    },
  }),
  columnHelper.accessor("n_trading_days", {
    header: "Days",
    cell: (info) => {
      const v = info.getValue();
      return <span className="font-mono">{v != null ? v : "—"}</span>;
    },
  }),
  columnHelper.accessor("n_windows", {
    header: "Windows",
    cell: (info) => {
      const v = info.getValue();
      return <span className="font-mono">{v != null ? v : "—"}</span>;
    },
  }),
];

export default function ExperimentsPage() {
  const { data, isLoading } = useExperiments();
  const [sorting, setSorting] = useState<SortingState>([
    { id: "sharpe_ratio", desc: true },
  ]);

  const tableData = useMemo(() => data || [], [data]);

  const table = useReactTable({
    data: tableData,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  if (isLoading) {
    return (
      <div className="space-y-6 max-w-7xl">
        <div>
          <h1 className="text-2xl font-bold">Experiments</h1>
          <p className="text-sm text-muted-foreground mt-1">Loading experiments...</p>
        </div>
        <div className="h-64 flex items-center justify-center text-muted-foreground">
          Loading...
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-7xl">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Experiments</h1>
          <p className="text-sm text-muted-foreground mt-1">
            {tableData.length} experiment{tableData.length !== 1 ? "s" : ""} — sorted by Sharpe Ratio
          </p>
        </div>
        <Link
          href="/lab"
          className="text-sm text-accent-foreground hover:underline"
        >
          ← Back to Lab
        </Link>
      </div>

      {tableData.length === 0 ? (
        <div className="rounded-lg border border-border bg-card p-12 text-center">
          <p className="text-muted-foreground">
            No experiments found · Run a walk-forward backtest to generate results
          </p>
        </div>
      ) : (
        <div className="rounded-lg border border-border overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              {table.getHeaderGroups().map((headerGroup) => (
                <tr key={headerGroup.id} className="border-b border-border bg-muted/50">
                  {headerGroup.headers.map((header) => (
                    <th
                      key={header.id}
                      className="px-4 py-3 text-left text-xs font-medium text-muted-foreground cursor-pointer select-none whitespace-nowrap"
                      onClick={header.column.getToggleSortingHandler()}
                    >
                      <div className="flex items-center gap-1">
                        {flexRender(header.column.columnDef.header, header.getContext())}
                        {header.column.getIsSorted() === "asc" && " ↑"}
                        {header.column.getIsSorted() === "desc" && " ↓"}
                      </div>
                    </th>
                  ))}
                </tr>
              ))}
            </thead>
            <tbody>
              {table.getRowModel().rows.map((row) => {
                const sharpe = row.original.sharpe_ratio;
                const rowBg =
                  sharpe != null && sharpe > 1.0
                    ? "bg-green-950/30"
                    : sharpe != null && sharpe < 0.5
                      ? "bg-red-950/30"
                      : "";
                return (
                  <tr key={row.id} className={cn("border-b border-border/50", rowBg)}>
                    {row.getVisibleCells().map((cell) => (
                      <td key={cell.id} className="px-4 py-3">
                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
