"""
loader.py — Data layer
======================
Tries Yahoo Finance first. Falls back to calibrated synthetic data
automatically if yfinance is unavailable or blocked.
Consistent with Papers 1-3: seed 42, Ledoit-Wolf shrinkage, 2019-2024.

Author : Mouheb Douzi  |  github.com/MouhabDouzi/cvar-factor-portfolio
"""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

TICKERS = [
    "AAPL","MSFT","NVDA","GOOGL","META",
    "JPM","BAC","GS","MS","BLK",
    "JNJ","UNH","PFE","MRK","ABBV",
    "XOM","CVX","COP","SLB","EOG",
    "WMT","HD","AMZN","COST","TGT",
    "PG","KO","PEP","MCD","NKE",
]
START      = "2019-01-01"
END        = "2024-12-31"
RF_ANNUAL  = 0.045
RF_DAILY   = RF_ANNUAL / 252
COST_DRAG  = 0.0002 / 252   # 2 bps round-trip, distributed as daily drag


# ── Real data ─────────────────────────────────────────────────────────────────

def _try_real(tickers, start, end):
    try:
        import yfinance as yf
        raw = yf.download(tickers, start=start, end=end,
                          auto_adjust=True, progress=False)["Close"]
        raw = raw.dropna(axis=1, thresh=int(0.9 * len(raw))).ffill().dropna()
        if len(raw) < 100:
            return None
        returns = np.log(raw / raw.shift(1)).dropna() - COST_DRAG
        return returns
    except Exception:
        return None


def _try_ff3_real(start, end):
    try:
        import pandas_datareader.data as web
        ff = web.DataReader("F-F_Research_Data_Factors_daily",
                            "famafrench", start, end)[0] / 100.0
        ff.index = pd.to_datetime(ff.index)
        return ff
    except Exception:
        return None


# ── Synthetic fallback ────────────────────────────────────────────────────────

def _synthetic_returns(n_days=1509, start=START):
    rng = np.random.default_rng(42)
    N   = len(TICKERS)

    # Sector block correlation
    C = np.full((N, N), 0.28)
    sz = N // 6
    for s in range(6):
        lo, hi = s*sz, (s+1)*sz if s < 5 else N
        C[lo:hi, lo:hi] = 0.65
    np.fill_diagonal(C, 1.0)
    ev = np.linalg.eigvalsh(C)
    if ev.min() < 0:
        C += (-ev.min() + 1e-6) * np.eye(N)
    L = np.linalg.cholesky(C)

    # GJR-GARCH vol path
    h = np.zeros(n_days); h[0] = 1e-6 / (1 - 0.09 - 0.88 - 0.02)
    eps = rng.standard_normal(n_days)
    for t in range(1, n_days):
        h[t] = max(1e-8, 1e-6 + (0.09 + 0.04*(eps[t-1]<0))*eps[t-1]**2 + 0.88*h[t-1])
    vol_path = np.sqrt(h) / np.sqrt(h).mean()

    # Fat-tailed shocks (t4)
    z = rng.standard_t(df=4, size=(n_days, N)) / np.sqrt(4/(4-2))
    z_c = z @ L.T

    daily_vols = np.array([
        .32,.28,.45,.27,.35,.25,.28,.30,.29,.24,
        .18,.22,.23,.21,.22,.28,.26,.31,.32,.33,
        .19,.22,.30,.21,.25,.17,.18,.18,.19,.23,
    ]) / np.sqrt(252)

    daily_mus = np.array([
        .18,.16,.35,.14,.17,.10,.09,.12,.11,.13,
        .08,.11,.07,.08,.10,.06,.07,.08,.05,.09,
        .10,.11,.15,.12,.08,.08,.07,.08,.09,.10,
    ]) / 252

    R = daily_mus + daily_vols * vol_path[:, None] * z_c
    # Inject market crash days
    for d in [60,61,62,310,311,850,851,1100]:
        if d < n_days:
            R[d] -= rng.uniform(0.02, 0.06, N)

    dates = pd.bdate_range(start, periods=n_days)
    return pd.DataFrame(R - COST_DRAG, index=dates, columns=TICKERS)


def _synthetic_ff3(index):
    rng = np.random.default_rng(42)
    n   = len(index)
    return pd.DataFrame({
        "Mkt-RF": rng.normal(0.00038, 0.012, n),
        "SMB":    rng.normal(0.00004, 0.006, n),
        "HML":    rng.normal(-0.00002, 0.006, n),
        "RF":     np.full(n, RF_DAILY),
    }, index=index)


# ── Public entry point ────────────────────────────────────────────────────────

def load_dataset():
    """
    Single public entry point.  Returns:
        returns   : (T × N) daily net return DataFrame
        ff3       : (T × 4) FF3 factor DataFrame
        cov       : (N × N) Ledoit-Wolf covariance
        mu_sample : (N,) sample mean returns
        tickers   : list[str]
        rf_daily  : float
        source    : 'real' | 'synthetic'
    """
    print("[Loader] Trying Yahoo Finance...", end=" ", flush=True)
    returns = _try_real(TICKERS, START, END)
    if returns is not None:
        print("OK (real data)")
        source = "real"
        ff3 = _try_ff3_real(START, END)
        if ff3 is None:
            print("[Loader] FF3 unavailable — using synthetic factors")
            common = returns.index
            ff3 = _synthetic_ff3(common)
        common  = returns.index.intersection(ff3.index)
        returns = returns.loc[common]
        ff3     = ff3.loc[common]
    else:
        print("blocked — using synthetic fallback")
        source  = "synthetic"
        returns = _synthetic_returns()
        ff3     = _synthetic_ff3(returns.index)

    lw  = LedoitWolf().fit(returns.values)
    cov = pd.DataFrame(lw.covariance_,
                       index=returns.columns, columns=returns.columns)

    print(f"[Loader] {source.upper()} | {len(returns.columns)} assets | "
          f"{len(returns)} days | "
          f"{returns.index[0].date()} → {returns.index[-1].date()}")

    return dict(
        returns   = returns,
        ff3       = ff3,
        cov       = cov,
        mu_sample = returns.mean(),
        tickers   = list(returns.columns),
        rf_daily  = RF_DAILY,
        source    = source,
    )
