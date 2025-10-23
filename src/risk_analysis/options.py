"""Option portfolio risk simulation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass(frozen=True)
class OptionContract:
    ticker: str
    option_type: str  # "call" or "put"
    quantity: float
    strike: float
    maturity: float  # expressed in years


def black_scholes_prices(
    spot: np.ndarray,
    strike: np.ndarray,
    rate: float,
    time_to_maturity: np.ndarray,
    volatility: np.ndarray,
    dividend_yield: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return Black-Scholes call and put prices."""

    spot = np.asarray(spot, dtype="float64")
    strike = np.asarray(strike, dtype="float64")
    tau = np.maximum(np.asarray(time_to_maturity, dtype="float64"), 1e-6)
    sigma = np.asarray(volatility, dtype="float64")

    forward = spot * np.exp((rate - dividend_yield) * tau)
    d1 = (np.log(spot / strike) + (rate - dividend_yield + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    disc_stock = spot * np.exp(-dividend_yield * tau)
    disc_strike = strike * np.exp(-rate * tau)

    call = disc_stock * norm.cdf(d1) - disc_strike * norm.cdf(d2)
    put = disc_strike * norm.cdf(-d2) - disc_stock * norm.cdf(-d1)
    return call, put


def option_portfolio_valuation(
    spot_prices: pd.Series,
    annual_volatility: pd.Series,
    contracts: Iterable[OptionContract],
    *,
    risk_free_rate: float = 0.04,
    dividend_yield: float = 0.0,
) -> pd.DataFrame:
    """Value each option contract and the total portfolio."""

    records = []
    for contract in contracts:
        if contract.ticker not in spot_prices:
            raise ValueError(f"Missing spot price for {contract.ticker}")

        S0 = float(spot_prices[contract.ticker])
        sigma = float(annual_volatility[contract.ticker])
        K = float(contract.strike)
        tau = float(contract.maturity)

        call, put = black_scholes_prices(S0, K, risk_free_rate, tau, sigma, dividend_yield)
        option_value = call if contract.option_type.lower() == "call" else put
        position_value = contract.quantity * option_value

        records.append(
            {
                "ticker": contract.ticker,
                "type": contract.option_type,
                "quantity": contract.quantity,
                "strike": contract.strike,
                "maturity": contract.maturity,
                "option_value": option_value,
                "position_value": position_value,
            }
        )

    df = pd.DataFrame(records).set_index("ticker")
    df.loc["Portfolio", "position_value"] = df["position_value"].sum()
    return df


def _calc_risk_measures(port_pnl: np.ndarray, confidence: float) -> Tuple[float, float]:
    threshold = np.quantile(port_pnl, 1 - confidence)
    var = -threshold
    tail = port_pnl <= threshold
    es = -port_pnl[tail].mean() if tail.any() else np.nan
    return var, es


def _calc_marginal_contributions(
    portfolio_var: float,
    portfolio_pnl: np.ndarray,
    instrument_pnl: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    eps = 0.001 * portfolio_var
    near_var = (portfolio_pnl >= -portfolio_var - eps) & (portfolio_pnl <= -portfolio_var + eps)
    tail = portfolio_pnl <= -portfolio_var

    if near_var.any():
        mvar = -instrument_pnl[near_var].mean(axis=0)
    else:
        mvar = np.full(instrument_pnl.shape[1], np.nan)

    if tail.any():
        mes = -instrument_pnl[tail].mean(axis=0)
    else:
        mes = np.full(instrument_pnl.shape[1], np.nan)

    return mvar, mes


def simulate_option_var(
    log_returns: pd.DataFrame,
    spot_prices: pd.Series,
    contracts: Iterable[OptionContract],
    *,
    horizon_days: int = 10,
    confidence: float = 0.99,
    n_bootstrap: int = 10_000,
    risk_free_rate: float = 0.04,
    dividend_yield: float = 0.0,
    random_state: int | None = 123,
) -> dict:
    """Bootstrap a distribution of option portfolio P&L and compute VaR/ES."""

    rng = np.random.default_rng(random_state)

    contracts_list: List[OptionContract] = list(contracts)
    tickers = [contract.ticker for contract in contracts_list]
    returns = log_returns[tickers].to_numpy()

    annual_vol = pd.Series(
        np.sqrt(log_returns.var(axis=0, ddof=1) * 252), index=log_returns.columns
    )

    maturity_adjustment = horizon_days / 252

    initial_values = option_portfolio_valuation(
        spot_prices, annual_vol, contracts_list, risk_free_rate=risk_free_rate, dividend_yield=dividend_yield
    )

    S0 = spot_prices[tickers].to_numpy(dtype="float64")
    sigmas = annual_vol[tickers].to_numpy(dtype="float64")
    strikes = np.array([c.strike for c in contracts_list], dtype="float64")
    maturities = np.array([c.maturity for c in contracts_list], dtype="float64")
    quantities = np.array([c.quantity for c in contracts_list], dtype="float64")
    option_types = np.array([c.option_type.lower() for c in contracts_list])

    initial_option_values = initial_values.loc[tickers, "option_value"].to_numpy(dtype="float64")

    # Bootstrap scenarios
    n_obs = returns.shape[0]
    pnl_matrix = np.zeros((n_bootstrap, len(contracts_list)))

    for i in range(n_bootstrap):
        idx = rng.integers(0, n_obs, size=(horizon_days,))
        horizon_returns = returns[idx, :].sum(axis=0)
        ST = S0 * np.exp(horizon_returns)
        remaining_maturity = np.maximum(maturities - maturity_adjustment, 1e-6)

        call_vals, put_vals = black_scholes_prices(
            ST,
            strikes,
            risk_free_rate,
            remaining_maturity,
            sigmas,
            dividend_yield,
        )
        option_values = np.where(option_types == "call", call_vals, put_vals)
        pnl_matrix[i, :] = quantities * (option_values - initial_option_values)

    portfolio_pnl = pnl_matrix.sum(axis=1)
    boot_var, boot_es = _calc_risk_measures(portfolio_pnl, confidence)
    mvar_boot, mes_boot = _calc_marginal_contributions(boot_var, portfolio_pnl, pnl_matrix)

    # Gaussian analytical approximation
    mean_returns = log_returns[tickers].mean(axis=0).to_numpy(dtype="float64")
    cov_returns = log_returns[tickers].cov().to_numpy(dtype="float64")
    mv_samples = rng.multivariate_normal(mean_returns * horizon_days, cov_returns * horizon_days, size=n_bootstrap)
    ST_mc = S0 * np.exp(mv_samples)

    call_vals, put_vals = black_scholes_prices(
        ST_mc,
        strikes,
        risk_free_rate,
        np.maximum(maturities - maturity_adjustment, 1e-6),
        sigmas,
        dividend_yield,
    )
    option_values_mc = np.where(option_types == "call", call_vals, put_vals)
    pnl_mc = quantities * (option_values_mc - initial_option_values)
    portfolio_pnl_mc = pnl_mc.sum(axis=1)
    gauss_var, gauss_es = _calc_risk_measures(portfolio_pnl_mc, confidence)

    mvar_gauss, mes_gauss = _calc_marginal_contributions(gauss_var, portfolio_pnl_mc, pnl_mc)

    return {
        "initial_values": initial_values,
        "bootstrap": {
            "VaR": boot_var,
            "ES": boot_es,
            "MarginalVaR": mvar_boot,
            "MarginalES": mes_boot,
            "portfolio_pnl": portfolio_pnl,
            "instrument_pnl": pnl_matrix,
        },
        "gaussian": {
            "VaR": gauss_var,
            "ES": gauss_es,
            "MarginalVaR": mvar_gauss,
            "MarginalES": mes_gauss,
            "portfolio_pnl": portfolio_pnl_mc,
            "instrument_pnl": pnl_mc,
        },
    }
