"""
turbulence.py — Financial Turbulence Index.

Implements the turbulence index from Kritzman & Li (2010) as used in
the Ensemble Strategy paper (Yang et al., 2020, Section 3.2):

    turbulence_t = (y_t - μ)  ·  Σ^{-1}  ·  (y_t - μ)^T

Where:
    y_t   = stock returns for current period t   (R^D)
    μ     = mean of historical returns           (R^D)
    Σ     = covariance of historical returns     (R^{D×D})

When turbulence_t exceeds a threshold, trading is halted:
    - All held shares are sold (liquidation)
    - No new buying until turbulence falls below threshold
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def compute_turbulence(
    df: pd.DataFrame,
    ticker_col: str = "tic",
    date_col: str = "date",
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Compute the turbulence index for each trading day.

    The turbulence at day t measures how unusual the cross-sectional
    return vector is relative to the full history up to t.

    Args:
        df: Price dataframe with columns [date, tic, close, ...].
        ticker_col: Column identifying each stock.
        date_col: Column identifying each date.
        price_col: Price column to compute returns from.

    Returns:
        DataFrame indexed by date with column 'turbulence'.
    """
    # Pivot to get a price matrix: rows=dates, cols=tickers
    price_pivot = df.pivot_table(
        index=date_col, columns=ticker_col, values=price_col
    )
    price_pivot = price_pivot.sort_index()

    # Daily returns
    returns = price_pivot.pct_change().dropna()

    dates = returns.index.tolist()
    turbulence_values = []

    # We need at least some historical data for μ and Σ
    # Use a minimum lookback of 252 trading days (~1 year)
    min_lookback = 252

    for i in range(len(dates)):
        if i < min_lookback:
            # Not enough history yet
            turbulence_values.append(0.0)
            continue

        # Historical returns up to (but not including) current day
        hist_returns = returns.iloc[:i]

        # Current day return vector
        current_return = returns.iloc[i].values.reshape(1, -1)

        # Historical mean and covariance
        mu = hist_returns.mean().values.reshape(1, -1)
        cov_matrix = hist_returns.cov().values

        # Regularize covariance for numerical stability
        cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-8

        try:
            cov_inv = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            # If singular, use pseudo-inverse
            cov_inv = np.linalg.pinv(cov_matrix)

        # turbulence_t = (y_t - μ) · Σ^{-1} · (y_t - μ)^T
        diff = current_return - mu
        turb = float(diff @ cov_inv @ diff.T)
        turbulence_values.append(turb)

    turb_df = pd.DataFrame({
        date_col: dates,
        "turbulence": turbulence_values,
    })
    return turb_df


def calculate_turbulence_threshold(
    turbulence_series: pd.Series,
    percentile: float = 90.0,
) -> float:
    """
    Calculate the turbulence threshold as a percentile of the
    historical turbulence distribution.

    The paper recommends using a high percentile (e.g., 90th or 99th)
    to trigger liquidation only during extreme market events.

    Args:
        turbulence_series: Series of turbulence values.
        percentile: Percentile to use as threshold (default 90).

    Returns:
        Turbulence threshold value.
    """
    # Exclude zero values (warmup period) from percentile calculation
    valid = turbulence_series[turbulence_series > 0]
    if len(valid) == 0:
        return float("inf")
    return float(np.percentile(valid, percentile))


def is_turbulent(
    current_turbulence: float,
    threshold: float,
) -> bool:
    """
    Determine if current market conditions are turbulent.

    Per the paper (Section 3.2): when turbulence_t > threshold,
    halt buying and sell all shares.

    Args:
        current_turbulence: Current turbulence index value.
        threshold: Turbulence threshold.

    Returns:
        True if market is turbulent (should liquidate).
    """
    return current_turbulence > threshold
