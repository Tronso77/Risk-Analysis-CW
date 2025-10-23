"""Shrinkage covariance estimators translated from MATLAB implementations."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def _as_matrix(data: pd.DataFrame | np.ndarray) -> Tuple[np.ndarray, list[str]]:
    if isinstance(data, pd.DataFrame):
        return data.to_numpy(dtype="float64"), list(map(str, data.columns))
    array = np.asarray(data, dtype="float64")
    if array.ndim != 2:
        raise ValueError("Input must be a 2-D array or DataFrame.")
    labels = [f"Asset_{i}" for i in range(array.shape[1])]
    return array, labels


def constant_correlation_shrinkage(
    returns: pd.DataFrame | np.ndarray,
) -> Tuple[pd.DataFrame, float]:
    """Ledoit-Wolf shrinkage towards a constant correlation matrix."""

    x, labels = _as_matrix(returns)
    t, n = x.shape
    if t < 2:
        raise ValueError("At least two observations are required.")

    x = x - x.mean(axis=0, keepdims=True)
    sample = (x.T @ x) / t

    variances = np.diag(sample)
    sqrt_var = np.sqrt(variances)
    corr = sample / np.outer(sqrt_var, sqrt_var)
    r_bar = (corr.sum() - n) / (n * (n - 1))

    prior = r_bar * np.outer(sqrt_var, sqrt_var)
    np.fill_diagonal(prior, variances)

    y = x**2
    phi_mat = (y.T @ y) / t - 2 * (x.T @ x) * sample / t + sample**2
    phi = float(phi_mat.sum())

    term1 = ((x**3).T @ x) / t
    helper = (x.T @ x) / t
    helper_diag = np.diag(helper)
    term2 = helper_diag[:, None] * sample
    term3 = helper * variances[:, None]
    term4 = variances[:, None] * sample
    theta_mat = term1 - term2 - term3 + term4
    np.fill_diagonal(theta_mat, 0.0)

    rho = float(np.trace(phi_mat) + r_bar * np.sum(np.outer(1 / sqrt_var, sqrt_var) * theta_mat))
    gamma = float(np.linalg.norm(sample - prior, ord="fro") ** 2)

    kappa = (phi - rho) / gamma
    shrinkage = max(0.0, min(1.0, kappa / t))
    sigma = shrinkage * prior + (1 - shrinkage) * sample

    sigma_df = pd.DataFrame(sigma, index=labels, columns=labels)
    return sigma_df, shrinkage


def single_index_shrinkage(
    asset_returns: pd.DataFrame | np.ndarray,
    market_returns: Iterable[float] | pd.Series | np.ndarray,
) -> Tuple[pd.DataFrame, float]:
    """Single-index shrinkage estimator following Ledoit-Wolf (2003)."""

    Y, labels = _as_matrix(asset_returns)
    Ymkt = np.asarray(list(market_returns), dtype="float64").reshape(-1, 1)

    if Y.shape[0] != Ymkt.shape[0]:
        raise ValueError("Asset and market returns must share the same number of rows.")

    T, N = Y.shape

    Y = Y - Y.mean(axis=0, keepdims=True)
    Ymkt = Ymkt - Ymkt.mean(axis=0, keepdims=True)

    sample = (Y.T @ Y) / T
    cov_mkt = (Y.T @ Ymkt) / T  # (N, 1)
    var_mkt = float((Ymkt.T @ Ymkt) / T)

    if var_mkt <= 0:
        raise ValueError("Market variance must be positive for shrinkage computation.")

    beta = (cov_mkt / var_mkt).flatten()
    residuals = Y - Ymkt @ beta[np.newaxis, :]
    res_var = (residuals**2).mean(axis=0)

    target = np.outer(beta, beta) * var_mkt + np.diag(res_var)

    Y2 = Y**2
    sample2 = (Y2.T @ Y2) / T
    pi_mat = sample2 - sample**2
    pihat = float(pi_mat.sum())

    gamma_hat = float(np.linalg.norm(sample - target, ord="fro") ** 2)

    rho_diag = float(np.trace(pi_mat))

    market_matrix = np.repeat(Ymkt, N, axis=1)
    temp = Y * market_matrix
    cov_mkt_matrix = np.tile(cov_mkt, (1, N))
    v1 = (Y2.T @ temp) / T - cov_mkt_matrix * sample
    cov_mkt_matrix_T = cov_mkt_matrix.T
    roff1 = (
        np.sum(v1 * cov_mkt_matrix_T) / var_mkt
        - np.sum(np.diag(v1) * cov_mkt.flatten()) / var_mkt
    )

    v3 = (temp.T @ temp) / T - var_mkt * sample
    cov_outer = cov_mkt @ cov_mkt.T
    roff3 = (
        np.sum(v3 * cov_outer) / (var_mkt**2)
        - np.sum(np.diag(v3) * (cov_mkt.flatten() ** 2)) / (var_mkt**2)
    )

    rho_off = 2 * roff1 - roff3
    rho_hat = rho_diag + rho_off

    kappa_hat = (pihat - rho_hat) / gamma_hat
    shrinkage = max(0.0, min(1.0, kappa_hat / T))
    sigma = shrinkage * target + (1 - shrinkage) * sample

    sigma_df = pd.DataFrame(sigma, index=labels, columns=labels)
    return sigma_df, shrinkage
