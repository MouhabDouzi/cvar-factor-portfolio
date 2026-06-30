"""
forecasting.py — Fama-French 3-Factor expected return forecasting
=================================================================
OLS regression per asset → factor-adjusted μ replacing noisy sample mean.

E[rᵢ] = rf + βᵢ_mkt·λ_mkt + βᵢ_smb·λ_smb + βᵢ_hml·λ_hml

Author : Mouheb Douzi  |  github.com/MouhabDouzi/cvar-factor-portfolio
"""

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import warnings
warnings.filterwarnings("ignore")


def run_ff3(ds: dict) -> dict:
    """
    Compute FF3 betas and factor-adjusted expected returns.
    Adds 'betas', 'mu_ff3', 'mu_comp' keys to dataset dict.
    """
    returns  = ds["returns"]
    ff3      = ds["ff3"]
    rf_daily = ds["rf_daily"]

    excess = returns.subtract(ff3["RF"], axis=0)
    X      = add_constant(ff3[["Mkt-RF", "SMB", "HML"]])

    betas = {}
    for ticker in returns.columns:
        y  = excess[ticker].dropna()
        Xa = X.loc[y.index]
        try:
            m = OLS(y, Xa).fit()
            betas[ticker] = {
                "alpha":    m.params["const"],
                "beta_mkt": m.params["Mkt-RF"],
                "beta_smb": m.params["SMB"],
                "beta_hml": m.params["HML"],
                "r2":       m.rsquared,
                "t_alpha":  m.tvalues["const"],
                "p_alpha":  m.pvalues["const"],
            }
        except Exception:
            betas[ticker] = dict(alpha=0, beta_mkt=1, beta_smb=0,
                                  beta_hml=0, r2=0, t_alpha=0, p_alpha=1)

    betas_df = pd.DataFrame(betas).T

    lam_mkt = ff3["Mkt-RF"].mean()
    lam_smb = ff3["SMB"].mean()
    lam_hml = ff3["HML"].mean()

    mu_ff3 = (rf_daily
              + betas_df["beta_mkt"] * lam_mkt
              + betas_df["beta_smb"] * lam_smb
              + betas_df["beta_hml"] * lam_hml)
    mu_ff3.name = "mu_ff3"

    sig = (betas_df["p_alpha"] < 0.05).sum()
    print(f"[FF3] Betas estimated for {len(betas_df)} assets | "
          f"Significant alphas (p<0.05): {sig}/{len(betas_df)}")
    print(f"[FF3] Factor premia (daily): "
          f"Mkt={lam_mkt:.5f}  SMB={lam_smb:.5f}  HML={lam_hml:.5f}")

    ds.update(betas=betas_df, mu_ff3=mu_ff3,
              mu_comp=pd.DataFrame({"sample": ds["mu_sample"],
                                    "ff3":    mu_ff3,
                                    "delta":  mu_ff3 - ds["mu_sample"]}))
    return ds
