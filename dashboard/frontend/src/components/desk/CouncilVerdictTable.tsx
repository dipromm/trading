"use client";

import { useMemo, useState } from "react";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  flexRender,
  createColumnHelper,
  type SortingState,
  type ColumnFiltersState,
} from "@tanstack/react-table";
import { useCouncilVerdict, type TradeDecision } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";

const ASSET_CLASS_MAP: Record<string, string> = {
  TLT: "Bond", IEF: "Bond",
  GLD: "Gold",
  XLU: "Defensive", XLP: "Defensive",
  EFA: "International",
};

function getAssetClass(ticker: string): string {
  return ASSET_CLASS_MAP[ticker] || "Equity";
}

const columnHelper = createColumnHelper<TradeDecision>();

const columns = [
  columnHelper.accessor("ticker", {
    header: "Ticker",
    cell: (info) => <span className="font-mono font-medium">{info.getValue()}</span>,
  }),
  columnHelper.accessor("action", {
    header: "Verdict",
    cell: (info) => {
      const v = info.getValue();
      return (
        <Badge variant={v === "BUY" ? "success" : v === "SELL" ? "destructive" : "secondary"}>
          {v}
        </Badge>
      );
    },
  }),
  columnHelper.accessor("agent_votes.matematico", {
    id: "matematico",
    header: "El Matemático",
    cell: (info) => {
      const val = info.getValue() as number;
      return (
        <div className="flex items-center gap-2 min-w-[100px]">
          <Progress value={val * 100} className="flex-1" />
          <span className="text-xs font-mono w-8 text-right">{val.toFixed(2)}</span>
        </div>
      );
    },
  }),
  columnHelper.accessor("agent_votes.analista", {
    id: "analista",
    header: "El Analista",
    cell: (info) => {
      const val = info.getValue() as number | null;
      if (val == null) return <span className="text-muted-foreground text-xs">—</span>;
      return (
        <div className="flex items-center gap-2 min-w-[100px]">
          <Progress value={val * 100} className="flex-1" />
          <span className="text-xs font-mono w-8 text-right">{val.toFixed(2)}</span>
        </div>
      );
    },
  }),
  columnHelper.accessor("agent_votes.cazador", {
    id: "cazador",
    header: "El Cazador",
    cell: (info) => {
      const val = info.getValue() as string;
      return (
        <Badge variant={val === "signal" ? "signal" : "silence"}>
          {val === "signal" ? "SIGNAL" : "SILENCE"}
        </Badge>
      );
    },
  }),
  columnHelper.accessor("agent_votes.conspiranoico", {
    id: "conspiranoico",
    header: "El Conspiranoico",
    cell: (info) => {
      const val = info.getValue() as string;
      const isActive = val === "active";
      return (
        <div className="flex items-center gap-1.5">
          <span className={cn("h-2.5 w-2.5 rounded-full", isActive ? "bg-red-500" : "bg-emerald-500")} />
          <span className="text-xs">{isActive ? "VETO" : "OK"}</span>
        </div>
      );
    },
  }),
  columnHelper.accessor("kelly_fraction", {
    header: "f* Kelly",
    cell: (info) => (
      <span className="font-mono text-right block">{info.getValue().toFixed(4)}</span>
    ),
  }),
];

export function CouncilVerdictTable() {
  const { data, isLoading } = useCouncilVerdict();
  const [sorting, setSorting] = useState<SortingState>([
    { id: "kelly_fraction", desc: true },
  ]);
  const [assetFilter, setAssetFilter] = useState<string>("all");

  const decisions = useMemo(() => {
    if (!data?.decisions) return [];
    if (assetFilter === "all") return data.decisions;
    return data.decisions.filter((d) => getAssetClass(d.ticker) === assetFilter);
  }, [data, assetFilter]);

  const table = useReactTable({
    data: decisions,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  if (isLoading) {
    return <div className="text-muted-foreground">Loading council verdict...</div>;
  }

  if (!data || data.status === "not_started") {
    return (
      <div className="rounded-lg border border-border bg-card p-8 text-center">
        <p className="text-muted-foreground">
          Paper trading not yet started · Walk-forward results available in the{" "}
          <a href="/lab" className="text-accent-foreground hover:underline">Lab</a>
        </p>
      </div>
    );
  }

  const isVetoActive = (data as { veto_active?: boolean }).veto_active === true;

  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">Council Verdict</h2>
        {data.date && (
          <span className="text-xs text-muted-foreground">{data.date}</span>
        )}
      </div>

      <Tabs value={assetFilter} onValueChange={setAssetFilter} className="mb-4">
        <TabsList>
          <TabsTrigger value="all">All</TabsTrigger>
          <TabsTrigger value="Equity">Equity</TabsTrigger>
          <TabsTrigger value="Bond">Bond</TabsTrigger>
          <TabsTrigger value="Gold">Gold</TabsTrigger>
          <TabsTrigger value="Defensive">Defensive</TabsTrigger>
        </TabsList>
      </Tabs>

      {isVetoActive && decisions.length === 0 && (
        <div className="mb-4 flex items-center gap-2 rounded-lg border border-red-900/50 bg-red-950/20 px-4 py-3 text-sm text-red-400">
          <span className="h-2.5 w-2.5 rounded-full bg-red-500 shrink-0" />
          <span>
            <strong>El Conspiranoico emitió veto</strong> — Régimen de riesgo elevado detectado.
            Todas las operaciones suspendidas para esta sesión.
          </span>
        </div>
      )}

      <div className="rounded-lg border border-border overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id} className="border-b border-border bg-muted/50">
                {headerGroup.headers.map((header) => (
                  <th
                    key={header.id}
                    className="px-3 py-3 text-left text-xs font-medium text-muted-foreground cursor-pointer select-none whitespace-nowrap"
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
              const action = row.original.action;
              const rowBg = action === "BUY"
                ? "bg-green-950/20"
                : action === "SELL"
                  ? "bg-red-950/20"
                  : "";
              return (
                <tr key={row.id} className={cn("border-b border-border/50", rowBg)}>
                  {row.getVisibleCells().map((cell) => (
                    <td key={cell.id} className="px-3 py-2.5">
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
