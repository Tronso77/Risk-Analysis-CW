"""Portfolio analytics translated from the original MATLAB scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.stats import jarque_bera, kurtosis, norm, skew, t


@dataclass
class PortfolioSummary:
    mean_return: float
    std_return: float
    skewness: float
    kurtosis: float
    min_return: float
    max_return: float
    jb_statistic: float
    jb_pvalue: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean_return": self.mean_return,
            "std_return": self.std_return,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "min_return": self.min_return,
            "max_return": self.max_return,
            "jb_statistic": self.jb_statistic,
            "jb_pvalue": self.jb_pvalue,
        }


def _normalize_weights(num_assets: int, weights: Iterable[float] | None) -> np.ndarray:
    if weights is None:
        return np.full(num_assets, 1.0 / num_assets)
    vector = np.asarray(list(weights), dtype="float64")
    if vector.shape != (num_assets,):
        raise ValueError(
            f"Weights must have length {num_assets}, received shape {vector.shape}."
        )
    if not np.isclose(vector.sum(), 1.0):
        vector = vector / vector.sum()
    return vector


def compute_portfolio_statistics(
    returns: pd.DataFrame,
    *,
    weights: Iterable[float] | None = None,
) -> Tuple[pd.Series, pd.DataFrame, PortfolioSummary]:
    """Return the portfolio series, asset statistics, and summary metrics."""

    if returns.empty:
        raise ValueError("Return table is empty.")

    weights_vector = _normalize_weights(returns.shape[1], weights)
    portfolio = returns @ weights_vector
    portfolio.name = "Portfolio"

    asset_stats = pd.DataFrame(
        {
            "mean": returns.mean(),
            "std": returns.std(ddof=1),
            "skew": returns.apply(lambda col: skew(col, bias=False)),
            "kurt": returns.apply(lambda col: kurtosis(col, fisher=False, bias=False)),
        }
    )

    jb_stat, jb_pvalue = jarque_bera(portfolio)
    summary = PortfolioSummary(
        mean_return=float(portfolio.mean()),
        std_return=float(portfolio.std(ddof=1)),
        skewness=float(skew(portfolio, bias=False)),
        kurtosis=float(kurtosis(portfolio, fisher=False, bias=False)),
        min_return=float(portfolio.min()),
        max_return=float(portfolio.max()),
        jb_statistic=float(jb_stat),
        jb_pvalue=float(jb_pvalue),
    )

    return portfolio, asset_stats, summary


def rolling_var_estimates(
    portfolio: pd.Series,
    *,
    window_months: int = 6,
    alphas: Tuple[float, float] = (0.90, 0.99),
    min_periods: int = 30,
    n_simulations: int = 10_000,
    random_state: int | None = 123,
) -> pd.DataFrame:
    """Compute a range of VaR estimators using a rolling 6-month window."""

    if portfolio.empty:
        raise ValueError("Portfolio series is empty.")

    rng = np.random.default_rng(random_state)
    records = []

    z_values = {alpha: norm.ppf(1 - alpha) for alpha in alphas}

    for date, _ in portfolio.iloc[min_periods:].items():
        window_start = date - pd.DateOffset(months=window_months)
        window = portfolio.loc[(portfolio.index >= window_start) & (portfolio.index < date)]

        if len(window) < min_periods:
            continue

        window_data = window.to_numpy()
        mu = window_data.mean()
        sigma = window_data.std(ddof=1)
        sample_skew = skew(window_data, bias=False)
        sample_kurt = kurtosis(window_data, fisher=False, bias=False)
        ex_kurt = sample_kurt - 3

        entry = {"Date": date}

        # Historical Simulation
        for alpha in alphas:
            key = f"HS_{int(alpha * 100):02d}"
            entry[key] = np.percentile(window_data, (1 - alpha) * 100)

        # Parametric normal and Monte Carlo normal
        for alpha in alphas:
            z = z_values[alpha]
            entry[f"Normal_{int(alpha * 100):02d}"] = mu + z * sigma

        mc_samples = rng.normal(mu, sigma, size=n_simulations)
        for alpha in alphas:
            entry[f"MC_{int(alpha * 100):02d}"] = np.quantile(mc_samples, 1 - alpha)

        # Cornish-Fisher expansion
        def cornish_fisher(z_value: float) -> float:
            return (
                z_value
                + (z_value**2 - 1) * sample_skew / 6
                + (z_value**3 - 3 * z_value) * ex_kurt / 24
            )

        for alpha in alphas:
            z_adj = cornish_fisher(z_values[alpha])
            entry[f"CF_{int(alpha * 100):02d}"] = mu + z_adj * sigma

        # Bootstrapped Historical Simulation
        boot_sample = rng.choice(window_data, size=len(window_data) * n_simulations, replace=True)
        for alpha in alphas:
            entry[f"BSHS_{int(alpha * 100):02d}"] = np.percentile(boot_sample, (1 - alpha) * 100)

        # Student-t (method of moments)
        if sample_kurt <= 3:
            for alpha in alphas:
                entry[f"T_{int(alpha * 100):02d}"] = np.nan
        else:
            nu = 4 + 6 / (sample_kurt - 3)
            if nu <= 2:
                for alpha in alphas:
                    entry[f"T_{int(alpha * 100):02d}"] = np.nan
            else:
                sample_var = window_data.var(ddof=1)
                sigma2_mm = ((nu - 2) / nu) * sample_var
                if sigma2_mm <= 0:
                    for alpha in alphas:
                        entry[f"T_{int(alpha * 100):02d}"] = np.nan
                else:
                    sigma_mm = np.sqrt(sigma2_mm)
                    for alpha in alphas:
                        tail_prob = 1 - alpha
                        entry[f"T_{int(alpha * 100):02d}"] = mu + sigma_mm * t.ppf(tail_prob, df=nu)

        records.append(entry)

    return pd.DataFrame.from_records(records).set_index("Date")


def summarize_var_violations(
    portfolio: pd.Series,
    var_table: pd.DataFrame,
) -> pd.DataFrame:
    """Estimate VaR violation rates for each method."""

    aligned_returns = portfolio.reindex(var_table.index).dropna()
    results = []
    for column in var_table.columns:
        var_series = var_table[column].dropna()
        aligned = aligned_returns.reindex(var_series.index).dropna()
        violations = (aligned < var_series.loc[aligned.index]).sum()
        total = aligned.shape[0]
        violation_fraction = violations / total if total else np.nan
        results.append(
            {
                "VaR_Method": column,
                "NumViolations": int(violations),
                "NumValidDays": int(total),
                "ViolationFraction": violation_fraction,
            }
        )

    return pd.DataFrame(results).set_index("VaR_Method")


def bootstrap_loss_probabilities(
    returns: pd.DataFrame,
    *,
    weights: Iterable[float] | None = None,
    loss_threshold: float = -0.05,
    max_horizon: int = 50,
    n_bootstrap: int = 10_000,
    random_state: int | None = 123,
    include_gaussian: bool = True,
) -> pd.DataFrame:
    """Estimate the probability of breaching a loss threshold across horizons."""

    if returns.empty:
        raise ValueError("Return table is empty.")

    weights_vector = _normalize_weights(returns.shape[1], weights)
    portfolio = (returns @ weights_vector).to_numpy()
    n_obs = portfolio.shape[0]

    rng = np.random.default_rng(random_state)
    boot_counts = np.zeros(max_horizon, dtype="float64")

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n_obs, size=n_obs)
        boot_sample = portfolio[indices]
        cumulative = np.cumsum(boot_sample[:max_horizon])
        boot_counts += (cumulative <= loss_threshold).astype("float64")

    boot_prob = boot_counts / n_bootstrap
    horizons = np.arange(1, max_horizon + 1)

    data = {"Bootstrapped": boot_prob}

    if include_gaussian:
        mu = portfolio.mean()
        sigma = portfolio.std(ddof=1)
        gaussian = norm.cdf(loss_threshold, loc=horizons * mu, scale=np.sqrt(horizons) * sigma)
        data["Gaussian"] = gaussian

    return pd.DataFrame(data, index=horizons)
