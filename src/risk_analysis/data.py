"""Data loading helpers for the RiskAnalysis notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
import numpy as np


def _candidate_paths(path: Path) -> Iterable[Path]:
    yield path
    if path.is_absolute():
        return

    cwd_path = Path.cwd() / path
    if cwd_path != path:
        yield cwd_path

    project_root = Path(__file__).resolve().parents[2]
    root_path = project_root / path
    if root_path != cwd_path:
        yield root_path

    assets_path = project_root / "assets" / path
    if assets_path not in {path, cwd_path, root_path}:
        yield assets_path


def load_price_history(
    filepath: str | Path = "cleaned_adj_close_data.xlsx",
    *,
    date_column: str = "Date",
    assets: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return a tidy price table indexed by dates.

    Parameters
    ----------
    filepath:
        Location of the Excel workbook (or CSV file). Relative paths are
        resolved from the repository root.
    date_column:
        Column name that stores the observation date.
    assets:
        Optional iterable of ticker names to keep. When omitted all columns
        are returned.
    """
    target = None
    original = Path(filepath)
    for candidate in _candidate_paths(original):
        if candidate.exists():
            target = candidate
            break

    if target is None:
        raise FileNotFoundError(f"Price file not found: {original}")

    if target.suffix.lower() == ".csv":
        raw = pd.read_csv(target)
    else:
        raw = pd.read_excel(target)

    if date_column not in raw.columns:
        raise ValueError(
            f"Expected a '{date_column}' column in {target.name},"
            f" but found {list(raw.columns)}"
        )

    price_cols = [c for c in raw.columns if c != date_column]
    if assets is not None:
        assets = list(assets)
        missing = sorted(set(assets) - set(price_cols))
        if missing:
            raise ValueError(f"Assets not present in file: {missing}")
        price_cols = [c for c in price_cols if c in assets]

    prices = (
        raw[[date_column, *price_cols]]
        .assign(Date=lambda df: pd.to_datetime(df[date_column]))
        .set_index(date_column)
        .sort_index()
    )
    prices.columns.name = "Ticker"
    return prices


def compute_log_returns(prices: pd.DataFrame) -> Tuple[pd.DatetimeIndex, pd.DataFrame]:
    """Compute daily log returns for the provided price table."""
    if prices.empty:
        raise ValueError("Price table is empty, cannot compute returns.")

    numeric_prices = prices.astype("float64")
    log_prices = np.log(numeric_prices)
    log_returns = log_prices.diff().dropna()
    log_returns.columns.name = "Ticker"
    return log_returns.index, log_returns
