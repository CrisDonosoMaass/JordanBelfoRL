"""
indicators.py — Extensible Technical Indicators Module.

Computes the technical indicators specified in the Ensemble Strategy paper
(Yang et al., 2020, Section 4.1.1):
    - MACD  (Moving Average Convergence Divergence)
    - RSI   (Relative Strength Index)
    - CCI   (Commodity Channel Index)
    - ADX   (Average Directional Index)

The module uses a registry pattern so users can add custom indicators
by decorating functions with @register_indicator.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import pandas as pd
import ta


# ---------------------------------------------------------------------------
# Indicator registry
# ---------------------------------------------------------------------------

_INDICATOR_REGISTRY: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {}


def register_indicator(name: str):
    """Decorator to register an indicator function by name."""
    def decorator(func: Callable[[pd.DataFrame], pd.Series]):
        _INDICATOR_REGISTRY[name] = func
        return func
    return decorator


def get_available_indicators() -> List[str]:
    """Return the list of all registered indicator names."""
    return list(_INDICATOR_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Default indicators from the paper (Section 4.1.1)
# ---------------------------------------------------------------------------

@register_indicator("macd")
def compute_macd(df: pd.DataFrame) -> pd.Series:
    """
    Moving Average Convergence Divergence.
    Uses standard 12/26/9 periods on the close price.
    This is the MACD histogram (signal line difference) as used in the paper.
    """
    indicator = ta.trend.MACD(
        close=df["close"],
        window_slow=26,
        window_fast=12,
        window_sign=9,
    )
    return indicator.macd()


@register_indicator("rsi")
def compute_rsi(df: pd.DataFrame) -> pd.Series:
    """
    Relative Strength Index (14 periods).
    Quantifies extent of recent price changes (overbought/oversold).
    """
    indicator = ta.momentum.RSIIndicator(close=df["close"], window=14)
    return indicator.rsi()


@register_indicator("cci")
def compute_cci(df: pd.DataFrame) -> pd.Series:
    """
    Commodity Channel Index (20 periods).
    Calculated using high, low, and close price.
    """
    indicator = ta.trend.CCIIndicator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=20,
    )
    return indicator.cci()


@register_indicator("adx")
def compute_adx(df: pd.DataFrame) -> pd.Series:
    """
    Average Directional Index (14 periods).
    Identifies trend strength using high, low, and close price.
    """
    indicator = ta.trend.ADXIndicator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14,
    )
    return indicator.adx()


# ---------------------------------------------------------------------------
# Extended indicators (extensible by users)
# ---------------------------------------------------------------------------

@register_indicator("bollinger_upper")
def compute_bollinger_upper(df: pd.DataFrame) -> pd.Series:
    """Upper Bollinger Band (20 periods, 2 std)."""
    indicator = ta.volatility.BollingerBands(
        close=df["close"], window=20, window_dev=2
    )
    return indicator.bollinger_hband()


@register_indicator("bollinger_lower")
def compute_bollinger_lower(df: pd.DataFrame) -> pd.Series:
    """Lower Bollinger Band (20 periods, 2 std)."""
    indicator = ta.volatility.BollingerBands(
        close=df["close"], window=20, window_dev=2
    )
    return indicator.bollinger_lband()


@register_indicator("ema_20")
def compute_ema_20(df: pd.DataFrame) -> pd.Series:
    """Exponential Moving Average (20 periods)."""
    indicator = ta.trend.EMAIndicator(close=df["close"], window=20)
    return indicator.ema_indicator()


@register_indicator("obv")
def compute_obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    indicator = ta.volume.OnBalanceVolumeIndicator(
        close=df["close"], volume=df["volume"]
    )
    return indicator.on_balance_volume()


@register_indicator("atr")
def compute_atr(df: pd.DataFrame) -> pd.Series:
    """Average True Range (14 periods)."""
    indicator = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=14
    )
    return indicator.average_true_range()


@register_indicator("vwap")
def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Volume Weighted Average Price.
    Simplified daily VWAP using typical price * volume / cumulative volume.
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()
    return vwap


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def add_indicators(
    df: pd.DataFrame,
    indicator_names: List[str],
    ticker_col: str = "tic",
) -> pd.DataFrame:
    """
    Add the requested technical indicators to the dataframe.

    The dataframe must have columns: close, high, low, volume, and a
    ticker identifier column. Indicators are computed per-ticker.

    Args:
        df: Price dataframe with OHLCV columns.
        indicator_names: List of indicator names to compute.
        ticker_col: Column name that identifies each ticker.

    Returns:
        DataFrame with indicator columns added.
    """
    df = df.copy()

    for ind_name in indicator_names:
        if ind_name not in _INDICATOR_REGISTRY:
            raise ValueError(
                f"Unknown indicator '{ind_name}'. "
                f"Available: {get_available_indicators()}"
            )
        df[ind_name] = 0.0

    # Compute per ticker to avoid cross-stock contamination
    result_frames = []
    for tic, group in df.groupby(ticker_col):
        group = group.sort_values("date").copy()
        for ind_name in indicator_names:
            compute_fn = _INDICATOR_REGISTRY[ind_name]
            group[ind_name] = compute_fn(group)
        result_frames.append(group)

    result = pd.concat(result_frames, ignore_index=True)
    result = result.sort_values(["date", ticker_col]).reset_index(drop=True)

    # Fill NaN from indicator warm-up periods
    for ind_name in indicator_names:
        result[ind_name] = result[ind_name].fillna(0.0)

    return result
