# RiskAnalysis 

Portfolio risk analysis project. It includes
 notebooks,  modules, and documentation.

## Project tour

- `notebooks/01_portfolio_risk.ipynb` — descriptive statistics, Jarque–Bera test,
  and a rolling Value-at-Risk (VaR) dashboard for an equally weighted equity
  portfolio.
- `notebooks/02_covariance_and_options.ipynb` — Ledoit–Wolf and single-index
  shrinkage estimators plus a Black–Scholes option portfolio VaR simulation.
- `notebooks/03_loss_probability.ipynb` — bootstrap vs. Gaussian estimates for
  the probability of exceeding a 5% drawdown across holding periods.
- `src/risk_analysis/` — Python translations of the MATLAB helpers (data loading,
  portfolio analytics, covariance shrinkage, and option pricing).
- Original MATLAB files (`q1.m`, `q4.m`, `q5_*.m`, etc.) are kept for reference.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/01_portfolio_risk.ipynb
```

The Excel and CSV datasets already live at the repository root. The notebooks
assume they remain alongside the project.
