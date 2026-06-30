"""
optimizer.py — CVaR + MVO portfolio optimizers
===============================================
CVaR via Rockafellar-Uryasev (2000) LP:
    min  γ + 1/(n(1-α)) · Σzᵢ
    s.t. zᵢ ≥ -rᵢ'w - γ,  zᵢ ≥ 0
         1'w=1, 0≤wⱼ≤0.15, sector ≤ 0.40

MVO via standard quadratic program (replicates Papers 1-3).

Author : Mouheb Douzi  |  github.com/MouhabDouzi/cvar-factor-portfolio
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

SECTOR_MAP = {
    "Tech":       ["AAPL","MSFT","NVDA","GOOGL","META"],
    "Financials": ["JPM","BAC","GS","MS","BLK"],
    "Healthcare": ["JNJ","UNH","PFE","MRK","ABBV"],
    "Energy":     ["XOM","CVX","COP","SLB","EOG"],
    "Consumer":   ["WMT","HD","AMZN","COST","TGT"],
    "Staples":    ["PG","KO","PEP","MCD","NKE"],
}
MAX_W      = 0.15
SECTOR_CAP = 0.40


@dataclass
class PortfolioResult:
    label:       str
    weights:     pd.Series
    exp_ret:     float    # annualized
    volatility:  float    # annualized
    sharpe:      float
    cvar:        float    # annualized CVaR (nan for MVO)
    alpha:       float    # CVaR confidence level (nan for MVO)
    status:      str


def _sector_cons(w, tickers, cap=SECTOR_CAP):
    cons = []
    for members in SECTOR_MAP.values():
        idx = [i for i, t in enumerate(tickers) if t in members]
        if idx:
            cons.append(cp.sum(w[idx]) <= cap)
    return cons


def optimize_cvar(returns: pd.DataFrame, mu: pd.Series,
                  alpha: float = 0.95,
                  rf_daily: float = 0.045/252) -> PortfolioResult:
    R, (n, N) = returns.values, returns.shape
    tickers   = list(returns.columns)
    w   = cp.Variable(N)
    gam = cp.Variable()
    z   = cp.Variable(n)

    obj  = gam + cp.sum(z) / (n * (1 - alpha))
    cons = [z >= -R @ w - gam, z >= 0,
            cp.sum(w) == 1, w >= 0, w <= MAX_W] + _sector_cons(w, tickers)

    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=cp.CLARABEL, verbose=False)
    if prob.status not in ["optimal","optimal_inaccurate"]:
        prob.solve(solver=cp.SCS, verbose=False)

    w_val = pd.Series(np.maximum(w.value, 0), index=tickers)
    w_val /= w_val.sum()
    mu_v   = mu.reindex(tickers).fillna(0).values
    Sigma  = returns.cov().values

    er  = float(w_val @ mu_v) * 252
    vol = float(np.sqrt(w_val @ Sigma @ w_val)) * np.sqrt(252)
    sr  = (er - rf_daily*252) / (vol + 1e-10)
    cvar_ann = float(obj.value) * np.sqrt(252) if obj.value else np.nan

    return PortfolioResult(
        label=f"CVaR-{int(alpha*100)}+FF3",
        weights=w_val, exp_ret=er, volatility=vol,
        sharpe=sr, cvar=cvar_ann, alpha=alpha, status=prob.status
    )


def optimize_mvo(returns: pd.DataFrame, mu: pd.Series,
                 cov: pd.DataFrame,
                 rf_daily: float = 0.045/252) -> PortfolioResult:
    tickers = list(returns.columns)
    N       = len(tickers)
    mu_v    = mu.reindex(tickers).fillna(0).values
    Sigma   = cov.reindex(index=tickers, columns=tickers).values

    w    = cp.Variable(N)
    cons = [cp.sum(w)==1, w>=0, w<=MAX_W] + _sector_cons(w, tickers)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), cons)
    prob.solve(solver=cp.CLARABEL, verbose=False)
    if w.value is None:
        prob.solve(solver=cp.SCS, verbose=False)

    w_val = pd.Series(np.maximum(w.value, 0), index=tickers)
    w_val /= w_val.sum()

    er  = float(w_val @ mu_v) * 252
    vol = float(np.sqrt(w_val @ Sigma @ w_val)) * np.sqrt(252)
    sr  = (er - rf_daily*252) / (vol + 1e-10)

    return PortfolioResult(
        label="MVO+FF3", weights=w_val,
        exp_ret=er, volatility=vol, sharpe=sr,
        cvar=float("nan"), alpha=float("nan"), status=prob.status
    )


def run_all_optimizations(ds: dict) -> dict:
    """Run CVaR-90/95/99 + MVO and store results in ds['portfolios']."""
    portfolios = {}
    for alpha in [0.90, 0.95, 0.99]:
        label = f"CVaR-{int(alpha*100)}+FF3"
        print(f"  [{label}] optimizing...", end=" ", flush=True)
        r = optimize_cvar(ds["returns"], ds["mu_ff3"], alpha, ds["rf_daily"])
        portfolios[label] = r
        print(f"ret={r.exp_ret:.2%}  vol={r.volatility:.2%}  "
              f"sharpe={r.sharpe:.4f}  CVaR={r.cvar:.2%}  [{r.status}]")

    for label, mu_key in [("MVO+FF3","mu_ff3"), ("MVO+Sample","mu_sample")]:
        print(f"  [{label}] optimizing...", end=" ", flush=True)
        r = optimize_mvo(ds["returns"], ds[mu_key], ds["cov"], ds["rf_daily"])
        r.label = label
        portfolios[label] = r
        print(f"ret={r.exp_ret:.2%}  vol={r.volatility:.2%}  "
              f"sharpe={r.sharpe:.4f}  [{r.status}]")

    ds["portfolios"] = portfolios
    return ds
