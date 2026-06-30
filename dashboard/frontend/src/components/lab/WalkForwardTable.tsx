"use client";

import { useMemo, useState } from "react";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  flexRender,
  createColumnHelper,
  type SortingState,
} from "@tanstack/react-table";
import { useWalkForwardWindows, type WalkForwardWindow } from "@/lib/api";
import { cn } from "@/lib/utils";

const columnHelper = createColumnHelper<WalkForwardWindow>();

const columns = [
  columnHelper.accessor("iteration", {
    header: "#",
    cell: (info) => info.getValue(),
  }),
  columnHelper.accessor("val_start", {
    header: "Start",
    cell: (info) => info.getValue(),
  }),
  columnHelper.accessor("val_end", {
    header: "End",
    cell: (info) => info.getValue(),
  }),
  columnHelper.accessor("sharpe_ratio", {
    header: "Sharpe",
    cell: (info) => (
      <span className="font-mono">{info.getValue().toFixed(3)}</span>
    ),
  }),
  columnHelper.accessor("total_return_pct", {
    header: "Return",
    cell: (info) => {
      const v = info.getValue();
      return (
        <span className={cn("font-mono", v >= 0 ? "text-emerald-400" : "text-red-400")}>
          {v >= 0 ? "+" : ""}{v.toFixed(1)}%
        </span>
      );
    },
  }),
  columnHelper.accessor("max_drawdown_pct", {
    header: "Max DD",
    cell: (info) => (
      <span className="font-mono text-red-400">{info.getValue().toFixed(1)}%</span>
    ),
  }),
  columnHelper.accessor("n_tickers", {
    header: "Tickers",
    cell: (info) => info.getValue(),
  }),
];

export function WalkForwardTable() {
  const { data, isLoading } = useWalkForwardWindows();
  const [sorting, setSorting] = useState<SortingState>([]);

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
    return <div className="text-muted-foreground">Loading walk-forward windows...</div>;
  }

  if (!data || data.length === 0) {
    return (
      <div className="rounded-lg border border-border bg-card p-8 text-center">
        <p className="text-muted-foreground text-sm">
          No walk-forward windows available · Run a backtest experiment to generate window results
        </p>
      </div>
    );
  }

  return (
    <div className="w-full">
      <h2 className="text-lg font-semibold mb-4">Walk-Forward Windows</h2>
      <div className="rounded-lg border border-border overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id} className="border-b border-border bg-muted/50">
                {headerGroup.headers.map((header) => (
                  <th
                    key={header.id}
                    className="px-4 py-3 text-left text-xs font-medium text-muted-foreground cursor-pointer select-none"
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
              const rowBg = sharpe > 1.0
                ? "bg-green-950/30"
                : sharpe < 0.5
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
    </div>
  );
}
