"""
Barrido de todos los perfiles Fase 6 con carry_positions (post-Exp10 fix).
Genera experiments/... y un resumen en reports/experiment_grid_carry.json
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Baseline + experimentos Fase 6 (orden lógico Exp1..Exp10)
PROFILES: list[str | None] = [
    None,  # baseline sin perfil
    "profiles/exp1_menos_friccion.yaml",
    "profiles/exp2_stop_loss_holgado.yaml",
    "profiles/exp1_exp2_combined.yaml",
    "profiles/exp3_conspiranoico_soft.yaml",
    "profiles/exp1_exp3_combined.yaml",
    "profiles/exp4_cazador_modulador.yaml",
    "profiles/exp1_exp3_exp4_combined.yaml",
    "profiles/exp5_mas_exposicion_normal.yaml",
    "profiles/exp1_exp3_exp5_combined.yaml",
    "profiles/exp6_juez_passthrough.yaml",
    "profiles/exp1_exp3_exp6_combined.yaml",
    "profiles/exp7_rotacion_defensivos.yaml",
    "profiles/exp1_exp3_exp7_combined.yaml",
    "profiles/exp8_veto_tuned.yaml",
    "profiles/exp1_exp3_exp8_combined.yaml",
    "profiles/exp1_exp3_exp8_p3_combined.yaml",
    "profiles/exp1_exp3_exp8_exp9_combined.yaml",
    "profiles/exp1_exp3_exp8_exp10_combined.yaml",
]

PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
RUN = ROOT / "run.py"


def _latest_result(experiments_dir: Path, prefix: str) -> Path | None:
    matches = sorted(experiments_dir.glob(f"{prefix}_*"), reverse=True)
    for p in matches:
        rf = p / "results.json"
        if rf.exists():
            return rf
    return None


def main() -> int:
    experiments_dir = ROOT / "experiments"
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    rows: list[dict] = []

    for i, profile in enumerate(PROFILES, 1):
        label = profile or "baseline (config base)"
        print(f"\n{'='*72}\n[{i}/{len(PROFILES)}] {label}\n{'='*72}", flush=True)

        cmd = [str(PYTHON), str(RUN), "--cazador", "--conspiranoico"]
        if profile:
            cmd.extend(["--profile", profile])

        # Snapshot experiment dirs before run
        before = set(experiments_dir.glob("fase6_*"))

        rc = subprocess.call(cmd, cwd=ROOT)
        if rc != 0:
            rows.append({"profile": label, "error": f"exit_code={rc}"})
            continue

        after = set(experiments_dir.glob("fase6_*"))
        new_dirs = sorted(after - before, reverse=True)
        results_path = None
        if new_dirs:
            results_path = new_dirs[0] / "results.json"
        if results_path is None or not results_path.exists():
            # fallback: latest by prefix
            prefix = "fase6_mat_cazador_conspiranoico"
            if profile:
                name = Path(profile).stem
                prefix = f"fase6_mat_cazador_conspiranoico_{name}"
            results_path = _latest_result(experiments_dir, prefix)

        if not results_path or not results_path.exists():
            rows.append({"profile": label, "error": "no results.json"})
            continue

        data = json.loads(results_path.read_text(encoding="utf-8"))
        mas = data["mas"]["metrics"]
        bh = data.get("buy_and_hold", {})
        row = {
            "profile": label,
            "experiment": data.get("experiment"),
            "dir": str(results_path.parent.name),
            "sharpe": mas["sharpe_ratio"],
            "return_pct": mas["total_return_pct"],
            "maxdd_pct": mas["max_drawdown_pct"],
            "calmar": mas.get("calmar_ratio"),
            "bh_sharpe": bh.get("sharpe_ratio"),
            "bh_return_pct": bh.get("total_return_pct"),
            "delta_sharpe_vs_bh": round(mas["sharpe_ratio"] - bh.get("sharpe_ratio", 0), 3),
            "iter_2022_return": next(
                (w["total_return_pct"] for w in data["mas"]["window_results"]
                 if w.get("val_start", "").startswith("2022")),
                None,
            ),
        }
        close = data["mas"].get("window_close_summary")
        if close:
            row["close_commission"] = close.get("total_commission_closing")
        rows.append(row)
        print(
            f"  -> Sharpe={row['sharpe']:.3f} | Ret={row['return_pct']:+.1f}% | "
            f"MaxDD={row['maxdd_pct']:.1f}% | vs B&H Sharpe={row['delta_sharpe_vs_bh']:+.3f}",
            flush=True,
        )

    out = reports_dir / "experiment_grid_carry.json"
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "carry_positions_default": True,
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResumen guardado en {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
