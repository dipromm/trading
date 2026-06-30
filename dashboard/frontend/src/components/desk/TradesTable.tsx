"use client";

import { useState, useMemo, useCallback } from "react";
import {
  useReactTable,
  getCoreRowModel,
  flexRender,
  createColumnHelper,
  type ExpandedState,
} from "@tanstack/react-table";
import { useTrades, type TradeDecision } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";

const columnHelper = createColumnHelper<TradeDecision>();

const columns = [
  columnHelper.accessor("date", {
    header: "Date",
    cell: (info) => <span className="font-mono text-xs">{info.getValue()}</span>,
  }),
  columnHelper.accessor("ticker", {
    header: "Ticker",
    cell: (info) => <span className="font-mono font-medium">{info.getValue()}</span>,
  }),
  columnHelper.accessor("action", {
    header: "Action",
    cell: (info) => {
      const v = info.getValue();
      return (
        <Badge variant={v === "BUY" ? "success" : v === "SELL" ? "destructive" : "secondary"}>
          {v}
        </Badge>
      );
    },
  }),
  columnHelper.accessor("kelly_fraction", {
    header: "f* Kelly",
    cell: (info) => <span className="font-mono">{info.getValue().toFixed(4)}</span>,
  }),
  columnHelper.accessor("reason", {
    header: "Reason",
    cell: (info) => <span className="text-xs text-muted-foreground">{info.getValue()}</span>,
  }),
  columnHelper.display({
    id: "votes",
    header: "Agent Votes",
    cell: (info) => {
      const row = info.row.original;
      const votes = row.agent_votes;
      if (!votes) return <span className="text-muted-foreground text-[10px]">—</span>;

      const mat = typeof votes.matematico === "number" ? votes.matematico : null;
      const ana = typeof votes.analista === "number" ? votes.analista : null;
      // cazador may be a boolean/int (0/1) from backtester or "signal"/"silence" from run_daily
      const caz = votes.cazador === "signal" || votes.cazador === 1 || votes.cazador === true;
      // conspiranoico may be absent from older backtester logs
      const con = votes.conspiranoico != null
        ? votes.conspiranoico === "active"
        : null;

      return (
        <div className="text-[10px] text-muted-foreground space-x-2">
          {mat != null && <span>M:{mat.toFixed(2)}</span>}
          {ana != null && <span>A:{ana.toFixed(2)}</span>}
          <span>C:{caz ? "SIG" : "—"}</span>
          {con !== null && <span>V:{con ? "🔴" : "🟢"}</span>}
        </div>
      );
    },
  }),
];

export function TradesTable() {
  const [offset, setOffset] = useState(0);
  const limit = 50;
  const { data, isLoading } = useTrades(limit, offset);

  const tableData = useMemo(() => data?.trades || [], [data]);

  const table = useReactTable({
    data: tableData,
    columns,
    getCoreRowModel: getCoreRowModel(),
  });

  const exportCSV = useCallback(() => {
    if (!data?.trades?.length) return;
    const headers = ["date", "ticker", "action", "kelly_fraction", "reason", "probability"];
    const rows = data.trades.map((t) =>
      [t.date, t.ticker, t.action, t.kelly_fraction, t.reason, t.probability].join(",")
    );
    const csv = [headers.join(","), ...rows].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "trades.csv";
    a.click();
    URL.revokeObjectURL(url);
  }, [data]);

  if (isLoading) {
    return <div className="text-muted-foreground">Loading trades...</div>;
  }

  if (!data || data.status === "not_started") {
    return (
      <div className="rounded-lg border border-border bg-card p-6 text-center">
        <p className="text-muted-foreground text-sm">No trades recorded yet.</p>
      </div>
    );
  }

  const totalPages = Math.ceil(data.total / limit);
  const currentPage = Math.floor(offset / limit) + 1;

  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">Trade Log</h2>
        <button
          onClick={exportCSV}
          className="text-xs border border-border rounded px-3 py-1.5 text-muted-foreground hover:bg-muted transition-colors"
        >
          Export CSV
        </button>
      </div>

      <div className="rounded-lg border border-border overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id} className="border-b border-border bg-muted/50">
                {headerGroup.headers.map((header) => (
                  <th key={header.id} className="px-3 py-3 text-left text-xs font-medium text-muted-foreground">
                    {flexRender(header.column.columnDef.header, header.getContext())}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.map((row) => (
              <tr key={row.id} className="border-b border-border/50">
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id} className="px-3 py-2.5">
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {data.total > limit && (
        <div className="flex items-center justify-between mt-3">
          <p className="text-xs text-muted-foreground">
            {data.total} total trades · Page {currentPage} of {totalPages}
          </p>
          <div className="flex gap-2">
            <button
              disabled={offset === 0}
              onClick={() => setOffset(Math.max(0, offset - limit))}
              className="text-xs border border-border rounded px-2 py-1 disabled:opacity-30"
            >
              ← Prev
            </button>
            <button
              disabled={offset + limit >= data.total}
              onClick={() => setOffset(offset + limit)}
              className="text-xs border border-border rounded px-2 py-1 disabled:opacity-30"
            >
              Next →
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
