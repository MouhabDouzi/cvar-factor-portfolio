"""
╔══════════════════════════════════════════════════════════════════════╗
║  PAPER 4 — CVaR-Factor Portfolio                                     ║
║  Beyond Variance: CVaR-Optimal Portfolio Construction with           ║
║  Factor-Adjusted Return Forecasts                                    ║
║                                                                      ║
║  Author : Mouheb Douzi                                               ║
║  GitHub : github.com/MouhabDouzi/cvar-factor-portfolio               ║
║  SSRN   : ssrn.com/author=MouhabDouzi                                ║
║  Seed   : np.random.seed(42)  |  S&P 500  |  2019–2024              ║
╚══════════════════════════════════════════════════════════════════════╝

RUN:   python main.py

Outputs
-------
  figures/fig1_normality.png      — Fat tails & normality violation
  figures/fig2_frontier.png       — CVaR vs MVO frontier + Sharpe bars
  figures/fig3_weights.png        — Portfolio weight comparison
  figures/fig4_equity.png         — Equity curves + drawdown
  figures/fig5_rolling_sharpe.png — Rolling 6-month Sharpe comparison
  paper/table1_scorecard.csv      — Full performance scorecard (SSRN Table 1)
  Console                         — Full VALIDATION BLOCK for each portfolio
"""

import os, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Ensure we resolve local modules even when run from another directory ──────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "paper"),   exist_ok=True)

from loader      import load_dataset
from forecasting import run_ff3
from optimizer   import run_all_optimizations
from figures     import generate_all, performance_table
from validation  import validate, print_block

# ── Banner ────────────────────────────────────────────────────────────────────
BANNER = """
╔══════════════════════════════════════════════════════════════════════╗
║  PAPER 4  |  CVaR-Factor Portfolio  |  Mouheb Douzi                 ║
║  github.com/MouhabDouzi/cvar-factor-portfolio                        ║
╚══════════════════════════════════════════════════════════════════════╝"""
print(BANNER)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 1 — DATA")
print("═"*60)
ds = load_dataset()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — FAMA-FRENCH 3-FACTOR FORECASTING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 2 — FAMA-FRENCH 3-FACTOR EXPECTED RETURNS")
print("═"*60)
ds = run_ff3(ds)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — PORTFOLIO OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 3 — PORTFOLIO OPTIMIZATION (CVaR-90/95/99 + MVO)")
print("═"*60)
ds = run_all_optimizations(ds)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — FIGURES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 4 — GENERATING ALL FIGURES")
print("═"*60)
figure_paths = generate_all(ds)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — PERFORMANCE SCORECARD
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 5 — PERFORMANCE SCORECARD")
print("═"*60)
perf = performance_table(ds["returns"], ds["portfolios"], ds["rf_daily"])
print("\n  Table 1 — Full Performance Scorecard")
print(perf.to_string())
out_csv = os.path.join(ROOT, "paper", "table1_scorecard.csv")
perf.to_csv(out_csv)
print(f"\n  Saved → {out_csv}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — STATISTICAL VALIDATION (PIVOT-STATISTICS AUDIT)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 6 — STATISTICAL VALIDATION  (Pivot-Statistics Audit)")
print("═"*60)

tickers = list(ds["returns"].columns)

def _pf(key, returns=ds["returns"], portfolios=ds["portfolios"]):
    w = portfolios[key].weights.reindex(tickers).fillna(0); w /= w.sum()
    return (returns @ w).values

s_cvar = _pf("CVaR-95+FF3")
s_mvo  = _pf("MVO+FF3")
s_bh   = ds["returns"].mean(axis=1).values
DATES  = ("2019-01-01", "2024-12-31")

# ── Block A: CVaR-95 vs Equal-Weight baseline ─────────────────────────────────
print("\n── A: CVaR-95+FF3 vs Equal-Weight ──────────────────────────────────────")
blk_a = validate(s_cvar, s_bh,
                  name="CVaR-95+FF3",
                  base_name="Equal-Weight Buy-and-Hold",
                  sample_type="IN-SAMPLE", dates=DATES)
print_block(blk_a)

# ── Block B: MVO vs Equal-Weight baseline ─────────────────────────────────────
print("\n── B: MVO+FF3 vs Equal-Weight ───────────────────────────────────────────")
blk_b = validate(s_mvo, s_bh,
                  name="MVO+FF3",
                  base_name="Equal-Weight Buy-and-Hold",
                  sample_type="IN-SAMPLE", dates=DATES)
print_block(blk_b)

# ── Block C: CVaR-95 vs MVO  [THE PRIMARY PAPER 4 TEST] ──────────────────────
print("\n── C: CVaR-95+FF3 vs MVO+FF3  [PRIMARY TEST] ───────────────────────────")
blk_c = validate(s_cvar, s_mvo,
                  name="CVaR-95+FF3",
                  base_name="MVO+FF3  (Papers 1-3 methodology)",
                  sample_type="IN-SAMPLE", dates=DATES)
print_block(blk_c)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 — FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 7 — FINAL SUMMARY")
print("═"*60)

pf = ds["portfolios"]
rows = {}
for key in ["CVaR-90+FF3","CVaR-95+FF3","CVaR-99+FF3","MVO+FF3","MVO+Sample"]:
    if key in pf:
        r = pf[key]
        rows[key] = {
            "Ann. Return": f"{r.exp_ret:.2%}",
            "Ann. Vol":    f"{r.volatility:.2%}",
            "Sharpe":      f"{r.sharpe:.4f}",
            "CVaR-95":     f"{r.cvar:.2%}" if not np.isnan(r.cvar) else "—",
            "Status":      r.status,
        }
summary = pd.DataFrame(rows).T
print("\n  Portfolio Summary")
print(summary.to_string())

print("\n" + "═"*60)
print("  All figures  →  ./figures/")
print("  Table 1      →  ./paper/table1_scorecard.csv")
print("  Data source  : " + ds["source"].upper())
print("  Seed         : np.random.seed(42)")
print("  Universe     : S&P 500  |  2019–2024")
print("═"*60)
print("\n  github.com/MouhabDouzi/cvar-factor-portfolio")
print("  ssrn.com/author=MouhabDouzi\n")
