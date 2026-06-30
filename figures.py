"""
figures.py — All publication figures for Paper 4
==================================================
Fig 1 : Normality violation (fat tails, skewness, JB rejection)
Fig 2 : CVaR frontier vs MVO frontier + Sharpe bar chart
Fig 3 : Portfolio weight comparison
Fig 4 : Equity curves + drawdown
Fig 5 : Rolling Sharpe comparison

Author : Mouheb Douzi  |  github.com/MouhabDouzi/cvar-factor-portfolio
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

FIG = "figures"
COLORS = dict(
    cvar90="#e74c3c", cvar95="#c0392b", cvar99="#922b21",
    mvo="#2980b9", baseline="#7f8c8d", accent="#27ae60", bh="#f39c12"
)
plt.rcParams.update({
    "figure.facecolor":"white","axes.facecolor":"#f8f9fa",
    "axes.grid":True,"grid.alpha":0.35,"font.family":"DejaVu Sans",
})


def _pct(ax, axis="x"):
    fmt = plt.FuncFormatter(lambda x,_: f"{x:.0%}")
    if axis=="x": ax.xaxis.set_major_formatter(fmt)
    else:         ax.yaxis.set_major_formatter(fmt)


# ── Figure 1 ──────────────────────────────────────────────────────────────────

def fig1_normality(returns: pd.DataFrame):
    port = returns.mean(axis=1); r = port.values
    fig  = plt.figure(figsize=(16,10))
    gs   = gridspec.GridSpec(2,3,figure=fig,hspace=0.42,wspace=0.35)

    # 1a histogram vs normal
    ax = fig.add_subplot(gs[0,0])
    ax.hist(r, bins=80, density=True, alpha=0.6, color=COLORS["mvo"])
    mu_n,sd_n = stats.norm.fit(r)
    x  = np.linspace(r.min(),r.max(),300)
    ax.plot(x, stats.norm.pdf(x,mu_n,sd_n), color=COLORS["cvar95"], lw=2)
    ax.set_title("Return Distribution vs Normal", fontweight="bold")
    ax.text(.05,.95,f"Excess κ: {stats.kurtosis(r):.2f}\nSkew: {stats.skew(r):.2f}",
            transform=ax.transAxes,va="top",fontsize=9,
            bbox=dict(boxstyle="round",facecolor="wheat",alpha=.7))

    # 1b QQ plot
    ax = fig.add_subplot(gs[0,1])
    (osm,osr),(sl,ic,_) = stats.probplot(r,dist="norm")
    ax.scatter(osm,osr,alpha=.4,s=8,color=COLORS["mvo"])
    ax.plot(osm,sl*np.array(osm)+ic,color=COLORS["cvar95"],lw=2)
    ax.set_title("Normal Q-Q Plot", fontweight="bold")
    _,jbp = stats.jarque_bera(r)
    ax.text(.05,.95,f"JB p-value:\n{jbp:.2e}",transform=ax.transAxes,
            va="top",fontsize=9,bbox=dict(boxstyle="round",facecolor="wheat",alpha=.7))

    # 1c rolling kurtosis
    ax = fig.add_subplot(gs[0,2])
    rk = port.rolling(63).apply(stats.kurtosis)
    rk.plot(ax=ax,color=COLORS["cvar95"],alpha=.8,lw=1)
    ax.axhline(0,color=COLORS["baseline"],lw=1.5,ls="--",label="Normal (κ=0)")
    ax.fill_between(rk.index,rk.values,0,where=(rk.values>0),
                    alpha=.2,color=COLORS["cvar95"])
    ax.set_title("Rolling 63-day Excess Kurtosis", fontweight="bold")
    ax.legend(fontsize=8)

    # 1d tail comparison
    ax = fig.add_subplot(gs[1,0])
    al = np.linspace(.01,.10,50)
    ax.plot(al*100, [-stats.norm.ppf(a,mu_n,sd_n) for a in al],
            color=COLORS["mvo"],lw=2,label="Normal VaR")
    ax.plot(al*100, [np.percentile(r,a*100)*-1 for a in al],
            color=COLORS["accent"],lw=2,label="Historical VaR")
    ax.plot(al*100, [-r[r<=np.percentile(r,a*100)].mean() for a in al],
            color=COLORS["cvar95"],lw=2,label="CVaR/ES")
    ax.set_title("Left Tail: Normal VaR underestimates", fontweight="bold")
    ax.set_xlabel("Confidence %"); ax.invert_xaxis(); ax.legend(fontsize=8)

    # 1e cross-asset skewness
    ax = fig.add_subplot(gs[1,1])
    sk = returns.apply(stats.skew)
    ax.hist(sk,bins=20,color=COLORS["mvo"],alpha=.7,edgecolor="white")
    ax.axvline(0,color=COLORS["cvar95"],lw=2,ls="--",label="Normal (skew=0)")
    ax.axvline(sk.mean(),color=COLORS["bh"],lw=2,label=f"Mean={sk.mean():.2f}")
    ax.set_title("Cross-Asset Skewness", fontweight="bold"); ax.legend(fontsize=8)

    # 1f JB rejection
    ax = fig.add_subplot(gs[1,2])
    jbp_all = returns.apply(lambda x: stats.jarque_bera(x.dropna())[1])
    cols    = [COLORS["cvar95"] if p<.05 else COLORS["accent"] for p in jbp_all]
    ax.bar(range(len(jbp_all)),-np.log10(jbp_all+1e-300),color=cols,alpha=.8)
    ax.axhline(-np.log10(.05),color=COLORS["bh"],lw=2,ls="--",label="p=0.05")
    rej = (jbp_all<.05).sum()
    ax.set_title(f"Jarque-Bera: {rej}/{len(jbp_all)} reject normality",
                 fontweight="bold"); ax.set_xticks([]); ax.legend(fontsize=8)

    fig.suptitle("Figure 1 — Why Variance Fails: Fat Tails & Normality Rejection "
                 "across S&P 500 2019–2024",fontsize=12,fontweight="bold",y=1.01)
    path = f"{FIG}/fig1_normality.png"
    fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig)
    print(f"[Fig 1] {path}")
    return path


# ── Figure 2 ──────────────────────────────────────────────────────────────────

def fig2_frontier(portfolios: dict):
    fig, axes = plt.subplots(1,2,figsize=(16,6))

    ax = axes[0]
    specs = [
        ("CVaR-90+FF3", COLORS["cvar90"], "CVaR-90%", "o"),
        ("CVaR-95+FF3", COLORS["cvar95"], "CVaR-95%", "s"),
        ("CVaR-99+FF3", COLORS["cvar99"], "CVaR-99%", "^"),
        ("MVO+FF3",     COLORS["mvo"],    "MVO",      "D"),
        ("MVO+Sample",  COLORS["baseline"],"MVO+SampleMu","P"),
    ]
    for key, c, lbl, mk in specs:
        if key in portfolios:
            r = portfolios[key]
            ax.scatter(r.volatility, r.exp_ret, s=250, color=c,
                       marker=mk, zorder=10,
                       label=f"{lbl}  (Sh={r.sharpe:.3f})")
    _pct(ax,"x"); _pct(ax,"y")
    ax.set_xlabel("Annualized Volatility"); ax.set_ylabel("Annualized Expected Return")
    ax.set_title("Figure 2a — Optimal Portfolios: CVaR vs MVO", fontweight="bold")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    labels = []; sharpes = []; colors = []
    for key, c, lbl, _ in specs:
        if key in portfolios:
            labels.append(lbl); sharpes.append(portfolios[key].sharpe); colors.append(c)
    bars = ax2.bar(labels, sharpes, color=colors, alpha=.85, edgecolor="white",lw=1.5)
    for bar, v in zip(bars, sharpes):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.005,
                 f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")
    ax2.set_title("Figure 2b — Sharpe Ratio Comparison", fontweight="bold")
    ax2.set_ylabel("Sharpe Ratio")
    ax2.set_ylim(min(0, min(sharpes))-.1, max(sharpes)+.15)

    fig.suptitle("CVaR-Optimal vs MVO — S&P 500 2019–2024 | FF3-Adjusted Returns | "
                 "Ledoit-Wolf Covariance", fontsize=11, fontweight="bold")
    path = f"{FIG}/fig2_frontier.png"
    fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig)
    print(f"[Fig 2] {path}"); return path


# ── Figure 3 ──────────────────────────────────────────────────────────────────

def fig3_weights(portfolios: dict):
    fig, axes = plt.subplots(1,2,figsize=(16,6))
    for ax, key, c, title in [
        (axes[0],"CVaR-95+FF3",COLORS["cvar95"],"CVaR-95 Portfolio Weights"),
        (axes[1],"MVO+FF3",    COLORS["mvo"],   "MVO Portfolio Weights"),
    ]:
        if key not in portfolios: continue
        w = portfolios[key].weights.sort_values().pipe(lambda s: s[s>0.001])
        bars = ax.barh(w.index, w.values, color=c, alpha=.8, edgecolor="white")
        for bar,v in zip(bars, w.values):
            ax.text(v+.001, bar.get_y()+bar.get_height()/2,
                    f"{v:.1%}", va="center", fontsize=8)
        _pct(ax,"x")
        r = portfolios[key]
        ax.text(.98,.02,f"Return: {r.exp_ret:.2%}\nVol: {r.volatility:.2%}\n"
                f"Sharpe: {r.sharpe:.4f}",transform=ax.transAxes,
                ha="right",va="bottom",fontsize=9,
                bbox=dict(boxstyle="round",facecolor="wheat",alpha=.7))
        ax.set_title(f"Figure 3 — {title}", fontweight="bold")

    fig.suptitle("Portfolio Weight Allocation: CVaR-95 vs MVO",
                 fontsize=12, fontweight="bold")
    path = f"{FIG}/fig3_weights.png"
    fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig)
    print(f"[Fig 3] {path}"); return path


# ── Figure 4 ──────────────────────────────────────────────────────────────────

def fig4_equity_drawdown(returns: pd.DataFrame, portfolios: dict):
    tickers = list(returns.columns)

    def pf_ret(key):
        w = portfolios[key].weights.reindex(tickers).fillna(0)
        w /= w.sum(); return returns @ w

    s_cvar = pf_ret("CVaR-95+FF3")
    s_mvo  = pf_ret("MVO+FF3")
    s_bh   = returns.mean(axis=1)

    def eq(r): return (1+r).cumprod()
    def dd(r):
        c=eq(r); return (c-c.cummax())/c.cummax()

    fig, axes = plt.subplots(2,1,figsize=(14,10),sharex=True)

    axes[0].plot(eq(s_cvar),color=COLORS["cvar95"],lw=2,label="CVaR-95+FF3")
    axes[0].plot(eq(s_mvo), color=COLORS["mvo"],   lw=2,label="MVO+FF3")
    axes[0].plot(eq(s_bh),  color=COLORS["bh"],lw=1.5,ls="--",
                 label="Equal-Weight Baseline",alpha=.7)
    axes[0].set_title("Figure 4a — Equity Curves", fontweight="bold")
    axes[0].set_ylabel("Cumulative Return")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"{x:.1f}×"))
    axes[0].legend(fontsize=10)

    axes[1].fill_between(dd(s_cvar).index,dd(s_cvar).values,0,
                         alpha=.4,color=COLORS["cvar95"],label="CVaR-95+FF3")
    axes[1].fill_between(dd(s_mvo).index, dd(s_mvo).values, 0,
                         alpha=.4,color=COLORS["mvo"],    label="MVO+FF3")
    axes[1].plot(dd(s_bh).index,dd(s_bh).values,color=COLORS["bh"],
                 lw=1,ls="--",alpha=.7,label="Equal-Weight")
    axes[1].set_title("Figure 4b — Drawdown", fontweight="bold")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"{x:.0%}"))
    axes[1].legend(fontsize=10)

    fig.suptitle("Out-of-Sample Performance — S&P 500 2019–2024",
                 fontsize=13,fontweight="bold")
    path = f"{FIG}/fig4_equity.png"
    fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig)
    print(f"[Fig 4] {path}"); return path


# ── Figure 5 ──────────────────────────────────────────────────────────────────

def fig5_rolling_sharpe(returns: pd.DataFrame, portfolios: dict,
                         rf_daily=0.045/252, window=126):
    tickers = list(returns.columns)

    def pf_ret(key):
        w = portfolios[key].weights.reindex(tickers).fillna(0)
        w /= w.sum(); return returns @ w

    def roll_sharpe(r):
        ex = r - rf_daily
        return (ex.rolling(window).mean() / r.rolling(window).std()) * np.sqrt(252)

    rs_cvar = roll_sharpe(pf_ret("CVaR-95+FF3"))
    rs_mvo  = roll_sharpe(pf_ret("MVO+FF3"))
    rs_bh   = roll_sharpe(returns.mean(axis=1))

    fig, ax = plt.subplots(figsize=(14,6))
    ax.plot(rs_cvar,color=COLORS["cvar95"],lw=2,label="CVaR-95+FF3")
    ax.plot(rs_mvo, color=COLORS["mvo"],   lw=2,label="MVO+FF3")
    ax.plot(rs_bh,  color=COLORS["bh"],lw=1.5,ls="--",alpha=.7,
            label="Equal-Weight Baseline")
    ax.axhline(0,color="black",lw=.8,ls=":")
    ax.fill_between(rs_cvar.index, rs_cvar.values, rs_mvo.values,
                    where=(rs_cvar.values > rs_mvo.values),
                    alpha=.15,color=COLORS["cvar95"],label="CVaR > MVO")
    ax.set_title(f"Figure 5 — Rolling {window}-Day Sharpe Ratio",fontweight="bold")
    ax.set_ylabel("Sharpe Ratio (ann.)"); ax.legend(fontsize=10)
    path = f"{FIG}/fig5_rolling_sharpe.png"
    fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig)
    print(f"[Fig 5] {path}"); return path


# ── Performance table ─────────────────────────────────────────────────────────

def performance_table(returns: pd.DataFrame, portfolios: dict,
                       rf_daily=0.045/252) -> pd.DataFrame:
    from scipy import stats as sp
    tickers = list(returns.columns)
    rows = {}
    for key, label in [("CVaR-95+FF3","CVaR-95+FF3"),
                        ("MVO+FF3","MVO+FF3"),
                        ("EQUAL","Equal-Weight")]:
        if key == "EQUAL":
            w = pd.Series(np.ones(len(tickers))/len(tickers), index=tickers)
        else:
            if key not in portfolios: continue
            w = portfolios[key].weights.reindex(tickers).fillna(0); w/=w.sum()
        r  = returns @ w
        c  = (1+r).cumprod(); mdd = ((c-c.cummax())/c.cummax()).min()
        sv = np.sort(r.values); cvar = -sv[:max(1,int(len(sv)*0.05))].mean()
        rows[label] = {
            "Ann. Return":    f"{r.mean()*252:.2%}",
            "Ann. Vol":       f"{r.std()*np.sqrt(252):.2%}",
            "Sharpe":         f"{(r.mean()-rf_daily)*252/(r.std()*np.sqrt(252)+1e-9):.4f}",
            "Max Drawdown":   f"{mdd:.2%}",
            "CVaR-95":        f"{cvar:.4%}",
            "Win Rate":       f"{(r>0).mean():.2%}",
            "Profit Factor":  f"{r[r>0].sum()/(-r[r<0].sum()+1e-9):.3f}",
            "Skewness":       f"{sp.skew(r.values):.3f}",
            "Excess Kurt.":   f"{sp.kurtosis(r.values):.3f}",
        }
    return pd.DataFrame(rows).T


# wrapper called from main
def generate_all(ds: dict):
    import os; os.makedirs(FIG, exist_ok=True)
    portfolios = ds["portfolios"]
    returns    = ds["returns"]
    rf         = ds["rf_daily"]

    # need scipy stats inside performance_table — import there
    p1 = fig1_normality(returns)
    p2 = fig2_frontier(portfolios)
    p3 = fig3_weights(portfolios)
    p4 = fig4_equity_drawdown(returns, portfolios)
    p5 = fig5_rolling_sharpe(returns, portfolios, rf)
    return [p1,p2,p3,p4,p5]
