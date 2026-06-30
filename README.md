# Paper 4 — CVaR-Factor Portfolio

**Beyond Variance: CVaR-Optimal Portfolio Construction with Factor-Adjusted Return Forecasts**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![CVXPY](https://img.shields.io/badge/Optimizer-CVXPY_CLARABEL-orange)](https://cvxpy.org)
[![SSRN](https://img.shields.io/badge/SSRN-Submitted-red)](https://ssrn.com/author=MouhabDouzi)

> Part of a four-paper series by Mouheb Douzi.
> Consistent across all papers: S&P 500 2019–2024 · `np.random.seed(42)` · Ledoit-Wolf shrinkage.

---

## One command

```bash
pip install -r requirements.txt
python main.py
```

That's it. Real S&P 500 data is fetched automatically. If yfinance is unavailable, calibrated synthetic data (Student-t fat tails, GJR-GARCH volatility clustering, sector correlations) is used as fallback — same schema, same pipeline.

---

## Research series

| # | Method | This paper adds |
|---|--------|----------------|
| 1 | Markowitz / CML | — |
| 2 | GJR-GARCH vol targeting | Dynamic σ overlay on MVO |
| 3 | Constrained MVO scorecard | Evaluation framework |
| **4** | **CVaR + FF3** | **Attacks the variance objective itself** |

---

## Core idea

Papers 1–3 minimise variance. Variance assumes normality. S&P 500 returns are not normal — fat tails, negative skewness, Jarque-Bera rejection across all assets. This paper replaces the objective:

**Variance → CVaR** (Conditional Value-at-Risk / Expected Shortfall)  
**Sample mean → Fama-French 3-Factor expected returns**

CVaR optimizer via Rockafellar-Uryasev (2000) linear program, solved by CVXPY/CLARABEL.

---

## Outputs

| File | Content |
|------|---------|
| `figures/fig1_normality.png` | Fat tails, QQ plot, JB rejection |
| `figures/fig2_frontier.png` | CVaR vs MVO frontier + Sharpe bars |
| `figures/fig3_weights.png` | Portfolio weight comparison |
| `figures/fig4_equity.png` | Equity curves + drawdown |
| `figures/fig5_rolling_sharpe.png` | Rolling 6-month Sharpe |
| `paper/table1_scorecard.csv` | SSRN Table 1 |
| Console | Full VALIDATION BLOCK per portfolio |

---

## Project structure (flat — one folder)

```
paper4/
├── main.py          ← single entry point
├── loader.py        ← data (real + synthetic fallback)
├── forecasting.py   ← FF3 factor betas + adjusted μ
├── optimizer.py     ← CVaR LP + MVO
├── figures.py       ← all 5 publication figures
├── validation.py    ← pivot statistics + VALIDATION BLOCK
├── requirements.txt
└── README.md
```

---

## References

- Rockafellar & Uryasev (2000). *Optimization of Conditional Value-at-Risk.* Journal of Risk.
- Fama & French (1993). *Common risk factors in stock and bond returns.* JFE.
- Ledoit & Wolf (2004). *Well-conditioned covariance estimator.* JMVA.
- Diamond & Boyd (2016). *CVXPY.* JMLR.

---

**Mouheb Douzi** · [github.com/MouhabDouzi](https://github.com/MouhabDouzi) · [ssrn.com/author=MouhabDouzi](https://ssrn.com/author=MouhabDouzi)
