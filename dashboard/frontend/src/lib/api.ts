import { useQuery } from "@tanstack/react-query";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function fetchApi<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  return res.json();
}

// ── Types ────────────────────────────────────────────────────────────────────

export interface EquityCurvePoint {
  date: string;
  portfolio_value: number;
  buyhold_value: number | null;
  sma_value: number | null;
  period: "walkforward" | "holdout" | "paper_trading";
}

export interface MetricsData {
  total_return_pct: number;
  annualized_return_pct: number;
  sharpe_ratio: number;
  max_drawdown_pct: number;
  calmar_ratio: number;
  win_rate_pct: number;
  n_trading_days: number;
  sortino_ratio?: number | null;
  profit_factor?: number | null;
}

export interface BaselineMetrics {
  total_return_pct?: number | null;
  annualized_return_pct?: number | null;
  sharpe_ratio?: number | null;
  max_drawdown_pct?: number | null;
  calmar_ratio?: number | null;
  win_rate_pct?: number | null;
  n_trading_days?: number | null;
}

export interface PeriodMetrics {
  mas: MetricsData;
  buy_and_hold: BaselineMetrics;
  sma_crossover: BaselineMetrics;
}

export interface WalkForwardWindow {
  iteration: number;
  val_start: string;
  val_end: string;
  total_return_pct: number;
  sharpe_ratio: number;
  max_drawdown_pct: number;
  n_tickers: number;
}

export interface ReliabilityPoint {
  mean_predicted: number;
  fraction_of_positives: number;
}

export interface ReliabilityData {
  status?: string;
  calibration_curve: ReliabilityPoint[];
  perfect_calibration: ReliabilityPoint[];
}

export interface RegistryStatus {
  experiment_id: string;
  trained_at?: string | null;
  val_sharpe: number;
  val_max_drawdown: number;
  holdout_start: string;
  last_run_at?: string | null;
  last_run_status?: string | null;
}

export interface MarketCloseData {
  last_market_close: string;
  pipeline_ran_at: string | null;
}

export interface KellyInputs {
  p: number;
  b: number;
  rho: number;
}

export interface AgentVotes {
  matematico: number;
  analista: number | null;
  cazador: "signal" | "silence";
  conspiranoico: "active" | "inactive";
}

export interface TradeDecision {
  date: string;
  ticker: string;
  action: "BUY" | "SELL" | "HOLD";
  probability: number;
  kelly_fraction: number;
  kelly_inputs: KellyInputs;
  agent_votes: AgentVotes;
  reason: string;
  position_size_eur: number;
}

export interface CouncilVerdictData {
  status?: string;
  date?: string;
  decisions?: TradeDecision[];
}

export interface TradesResponse {
  status?: string;
  total: number;
  offset: number;
  limit: number;
  trades: TradeDecision[];
}

export interface PositionEntry {
  ticker: string;
  quantity: number;
  entry_price: number;
  entry_date: string;
  current_value?: number | null;
  allocated_eur?: number | null;
  unrealized_pnl?: number | null;
}

export interface PositionsData {
  status?: string;
  positions: PositionEntry[];
  total_value: number;
  cash: number;
  as_of?: string | null;
}

export interface ExperimentSummary {
  experiment_id: string;
  timestamp?: string | null;
  total_return_pct?: number | null;
  annualized_return_pct?: number | null;
  sharpe_ratio?: number | null;
  max_drawdown_pct?: number | null;
  calmar_ratio?: number | null;
  win_rate_pct?: number | null;
  n_trading_days?: number | null;
  n_windows?: number | null;
}

// ── Hooks ────────────────────────────────────────────────────────────────────

export function useEquityCurve() {
  return useQuery<EquityCurvePoint[]>({
    queryKey: ["equity-curve"],
    queryFn: () => fetchApi("/api/historical/equity-curve"),
  });
}

export function useMetrics(period?: "walkforward" | "holdout") {
  const path = period
    ? `/api/historical/metrics?period=${period}`
    : "/api/historical/metrics";
  return useQuery<PeriodMetrics | MetricsData>({
    queryKey: ["metrics", period],
    queryFn: () => fetchApi(path),
  });
}

export function useWalkForwardWindows() {
  return useQuery<WalkForwardWindow[]>({
    queryKey: ["walkforward-windows"],
    queryFn: () => fetchApi("/api/historical/walkforward-windows"),
  });
}

export function useReliabilityDiagram() {
  return useQuery<ReliabilityData>({
    queryKey: ["reliability-diagram"],
    queryFn: () => fetchApi("/api/historical/reliability-diagram"),
  });
}

export function useStatus() {
  return useQuery<RegistryStatus & { status?: string }>({
    queryKey: ["status"],
    queryFn: () => fetchApi("/api/live/status"),
    refetchInterval: 60_000,
  });
}

export function useMarketClose() {
  return useQuery<MarketCloseData>({
    queryKey: ["market-close"],
    queryFn: () => fetchApi("/api/live/market-close"),
  });
}

export function useCouncilVerdict() {
  return useQuery<CouncilVerdictData>({
    queryKey: ["council-verdict"],
    queryFn: () => fetchApi("/api/live/council-verdict"),
    refetchInterval: 60_000,
  });
}

export function useTrades(limit = 50, offset = 0) {
  return useQuery<TradesResponse>({
    queryKey: ["trades", limit, offset],
    queryFn: () => fetchApi(`/api/live/trades?limit=${limit}&offset=${offset}`),
  });
}

export function usePositions() {
  return useQuery<PositionsData>({
    queryKey: ["positions"],
    queryFn: () => fetchApi("/api/live/positions"),
    refetchInterval: 60_000,
  });
}

export function useExperiments() {
  return useQuery<ExperimentSummary[]>({
    queryKey: ["experiments"],
    queryFn: () => fetchApi("/api/experiments/"),
  });
}
