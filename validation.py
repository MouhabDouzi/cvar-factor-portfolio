"""
validation.py — Pivot statistics + VALIDATION BLOCK
=====================================================
Implements all pivot statistics from the university inference-statistics
formula sheet and auto-generates the full VALIDATION BLOCK report.

Pivots: one-sample t, paired t, chi-square variance CI,
        F-ratio, proportion z, difference of proportions z.

Author : Mouheb Douzi  |  github.com/MouhabDouzi/cvar-factor-portfolio
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class PivotResult:
    name:      str
    stat:      float
    p:         float
    ci_lo:     float
    ci_hi:     float
    n_eff:     int
    verdict:   str    # VALIDATED / INCONCLUSIVE / REJECTED
    detail:    str


def _verdict(lo, hi, null=0.0):
    if lo > null:               return "VALIDATED"
    if hi < null:               return "REJECTED"
    return "INCONCLUSIVE"


def _n_eff(n, h=1):
    return max(1, n // max(1, h))


# ── Pivot 2: one-sample t (main workhorse) ────────────────────────────────────

def t_mean(x, mu0=0.0, alpha=0.05, alt="greater", horizon=1):
    x    = np.asarray(x, float)
    n    = _n_eff(len(x), horizon)
    xb   = x.mean(); S = x.std(ddof=1); se = S / np.sqrt(n); t = (xb-mu0)/max(se,1e-12)
    df   = n - 1
    if alt == "greater":
        p = 1 - stats.t.cdf(t, df)
        lo = xb - stats.t.ppf(1-alpha, df)*se; hi = np.inf
    elif alt == "less":
        p = stats.t.cdf(t, df)
        lo = -np.inf; hi = xb + stats.t.ppf(1-alpha, df)*se
    else:
        p = 2*(1-stats.t.cdf(abs(t), df))
        tc = stats.t.ppf(1-alpha/2, df)
        lo = xb - tc*se; hi = xb + tc*se
    v = _verdict(lo, hi if hi!=np.inf else xb, mu0)
    return PivotResult("one-sample t", round(t,4), round(p,4),
                       round(lo,8), round(hi,8), n, v,
                       f"x̄={xb:.6f} S={S:.6f} t={t:.4f} df={df} p={p:.4f} "
                       f"CI=[{lo:.6f}, {'∞' if hi==np.inf else round(hi,6)}]")


# ── Pivot 7: paired t (model vs baseline) ────────────────────────────────────

def paired_t(model, baseline, alpha=0.05, alt="greater", horizon=1):
    return t_mean(np.asarray(model)-np.asarray(baseline),
                  mu0=0.0, alpha=alpha, alt=alt, horizon=horizon)


# ── Pivot 3: chi-square variance CI ──────────────────────────────────────────

def chi2_var_ci(x, alpha=0.05):
    x  = np.asarray(x, float); n = len(x); S2 = x.var(ddof=1); df = n-1
    lo_v = df*S2 / stats.chi2.ppf(1-alpha/2, df)
    hi_v = df*S2 / stats.chi2.ppf(alpha/2,   df)
    lo_s = np.sqrt(lo_v); hi_s = np.sqrt(hi_v)
    ratio = hi_s / (lo_s + 1e-12)
    v = "VALIDATED" if ratio<1.3 else ("INCONCLUSIVE" if ratio<2 else "REJECTED")
    return PivotResult("chi-square variance CI", round(float(S2),10), float("nan"),
                       round(lo_s,6), round(hi_s,6), n, v,
                       f"S²={S2:.2e} vol_CI=[{lo_s:.4%},{hi_s:.4%}] "
                       f"width_ratio={ratio:.2f}")


# ── Pivot 8: F-ratio variance test ────────────────────────────────────────────

def f_ratio(x, y, alpha=0.05, alt="two-sided"):
    x = np.asarray(x,float); y = np.asarray(y,float)
    n,m = len(x),len(y)
    Sx2,Sy2 = x.var(ddof=1), y.var(ddof=1)
    F = Sx2 / max(Sy2, 1e-12)
    df1,df2 = n-1, m-1
    if alt == "greater":
        p = 1-stats.f.cdf(F,df1,df2)
        lo = F/stats.f.ppf(1-alpha,df1,df2); hi = np.inf
    elif alt == "less":
        p = stats.f.cdf(F,df1,df2)
        lo = 0; hi = F/stats.f.ppf(alpha,df1,df2)
    else:
        p = 2*min(stats.f.cdf(F,df1,df2), 1-stats.f.cdf(F,df1,df2))
        lo = F/stats.f.ppf(1-alpha/2,df1,df2)
        hi = F/stats.f.ppf(alpha/2,  df1,df2)
    v = "VALIDATED" if p < alpha else "INCONCLUSIVE"
    return PivotResult("F-ratio variance", round(F,4), round(p,4),
                       round(lo,4), round(hi if hi!=np.inf else F*10,4),
                       n+m, v,
                       f"Sx={np.sqrt(Sx2*252):.2%} Sy={np.sqrt(Sy2*252):.2%} "
                       f"F={F:.4f} p={p:.4f}")


# ── Pivot 9: single proportion (win-rate vs WR*) ─────────────────────────────

def prop_test(wins, n, wr_star=0.50, alpha=0.05, alt="greater"):
    p  = wins/n; se = np.sqrt(p*(1-p)/n + 1e-12); z = (p-wr_star)/se
    if alt == "greater":
        pv = 1-stats.norm.cdf(z); lo = p-stats.norm.ppf(1-alpha)*se; hi = 1.0
    else:
        pv = 2*(1-stats.norm.cdf(abs(z)))
        zc = stats.norm.ppf(1-alpha/2); lo = p-zc*se; hi = p+zc*se
    v = _verdict(lo-wr_star, hi-wr_star, 0.0)
    return PivotResult("proportion (win-rate)", round(z,4), round(pv,4),
                       round(lo,4), round(hi,4), n, v,
                       f"WR={p:.2%} WR*={wr_star:.2%} z={z:.4f} p={pv:.4f} "
                       f"CI=[{lo:.2%},{hi:.2%}]")


def breakeven_wr(avg_win, avg_loss):
    L = abs(avg_loss)
    return L / (avg_win + L + 1e-12)


# ── VALIDATION BLOCK ──────────────────────────────────────────────────────────

SEP  = "═"*70
SEP2 = "─"*70

def validate(strategy: np.ndarray, baseline: np.ndarray,
             name: str = "Strategy", base_name: str = "Baseline",
             sample_type: str = "OOS", dates: tuple = ("2019","2024"),
             alpha: float = 0.05, horizon: int = 1) -> dict:
    """Run all pivot tests and return structured block dict."""
    s = np.asarray(strategy, float)
    b = np.asarray(baseline, float)

    # Economics
    xb   = s.mean(); S = s.std(ddof=1)
    cum  = pd.Series((1+s).cumprod()); peak = cum.cummax()
    mdd  = ((cum-peak)/peak).min()
    cvar = -np.sort(s)[:max(1,int(len(s)*0.05))].mean()
    wins = (s>0).sum(); n = len(s)
    aw   = s[s>0].mean() if wins>0 else 0.0
    al   = s[s<0].mean() if (n-wins)>0 else 0.0
    wr   = wins/n; wrs = breakeven_wr(aw, al)
    pf   = s[s>0].sum() / (-s[s<0].sum()+1e-12) if (s<0).any() else np.inf

    # Tests
    t1   = t_mean(s, mu0=0.0, alpha=alpha, alt="greater", horizon=horizon)
    t2   = paired_t(s, b, alpha=alpha, alt="greater", horizon=horizon)
    chi  = chi2_var_ci(s, alpha=alpha)
    wr_t = prop_test(int(wins), n, wr_star=wrs, alpha=alpha)
    f_t  = f_ratio(s, b, alpha=alpha)

    verdicts = [t1.verdict, t2.verdict]
    overall  = ("REJECTED"     if "REJECTED" in verdicts else
                "VALIDATED"    if all(v=="VALIDATED" for v in verdicts) else
                "INCONCLUSIVE")

    return dict(
        name=name, base_name=base_name, sample_type=sample_type,
        dates=dates, n=n, n_eff=t1.n_eff, horizon=horizon, alpha_sig=alpha,
        xb=xb, S=S, ann_ret=xb*252, ann_vol=S*np.sqrt(252),
        sharpe=(xb/(S+1e-12))*np.sqrt(252), mdd=mdd,
        cvar95=cvar, wr=wr, wr_star=wrs, aw=aw, al=al, pf=pf,
        t1=t1, t2=t2, chi=chi, wr_t=wr_t, f_t=f_t,
        overall=overall,
        generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )


def print_block(blk: dict):
    b = blk
    lines = [
        SEP, "  VALIDATION BLOCK", SEP,
        f"  Strategy    : {b['name']}",
        f"  Baseline    : {b['base_name']}",
        f"  Sample type : {b['sample_type']}",
        f"  Date range  : {b['dates'][0]} → {b['dates'][1]}",
        f"  Raw n / Eff n : {b['n']} / {b['n_eff']}  (horizon={b['horizon']})",
        f"  Generated   : {b['generated']}",
        SEP2, "  ECONOMIC SUMMARY", SEP2,
        f"  Ann. Return : {b['ann_ret']:.2%}",
        f"  Ann. Vol    : {b['ann_vol']:.2%}",
        f"  Sharpe      : {b['sharpe']:.4f}",
        f"  Max Drawdown: {b['mdd']:.2%}",
        f"  CVaR-95     : {b['cvar95']:.4%}",
        f"  Win Rate    : {b['wr']:.2%}  (WR* = {b['wr_star']:.2%})",
        f"  Profit Factor: {b['pf']:.4f}",
        SEP2, "  PIVOT TESTS", SEP2,
        f"  [1] Mean>0      │ {b['t1'].detail}",
        f"      Verdict     │ {b['t1'].verdict}",
        "",
        f"  [2] Paired t    │ {b['t2'].detail}",
        f"      Verdict     │ {b['t2'].verdict}",
        "",
        f"  [3] Win-rate    │ {b['wr_t'].detail}",
        f"      Verdict     │ {b['wr_t'].verdict}",
        "",
        f"  [4] Var CI      │ {b['chi'].detail}",
        f"      Verdict     │ {b['chi'].verdict}",
        "",
        f"  [5] F-ratio     │ {b['f_t'].detail}",
        f"      Verdict     │ {b['f_t'].verdict}",
        SEP,
        f"  OVERALL VERDICT  :  >>>  {b['overall']}  <<<",
        SEP,
    ]
    print("\n".join(lines))
