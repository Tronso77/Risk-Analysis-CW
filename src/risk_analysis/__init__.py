"""Core package for the RiskAnalysis project.

This package exposes helper functions used by the example notebooks.
"""

from .data import load_price_history  # noqa: F401
from .portfolio import (
    bootstrap_loss_probabilities,
    compute_portfolio_statistics,
    rolling_var_estimates,
    summarize_var_violations,
)  # noqa: F401
from .shrinkage import (
    constant_correlation_shrinkage,
    single_index_shrinkage,
)  # noqa: F401
from .options import (
    option_portfolio_valuation,
    simulate_option_var,
)  # noqa: F401

__all__ = [
    "load_price_history",
    "bootstrap_loss_probabilities",
    "compute_portfolio_statistics",
    "rolling_var_estimates",
    "summarize_var_violations",
    "constant_correlation_shrinkage",
    "single_index_shrinkage",
    "option_portfolio_valuation",
    "simulate_option_var",
]
