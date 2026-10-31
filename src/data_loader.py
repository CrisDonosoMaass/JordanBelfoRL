"""
data_loader.py — Data Download, Cleaning, and Preprocessing Module.

Handles:
    1. Downloading OHLCV data from yfinance for the configured tickers
    2. Data cleaning (NaN handling, sorting, type coercion)
    3. Computing technical indicators via indicators.py
    4. Computing turbulence index via turbulence.py
    5. Caching processed data to disk (CSV/parquet)
    6. Incremental updates (download fresh data to current date)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from src.indicators import add_indicators
from src.turbulence import compute_turbulence

logger = logging.getLogger("drl_ensemble.data_loader")


# ---------------------------------------------------------------------------
# Main DataLoader class
# ---------------------------------------------------------------------------

class DataLoader:
    """
    Downloads, cleans, enriches, and caches market data.

    Attributes:
        tickers: List of stock tickers.
        start_date: Earliest date to download.
        end_date: Latest date to download.
        indicator_names: Technical indicators to compute.
        data_dir: Directory for caching data files.
    """

    def __init__(self, config: dict):
        self.tickers: List[str] = config["tickers"]
        self.start_date: str = config["dates"]["train_start"]
        self.end_date: str = config["dates"]["trade_end"]
        self.indicator_names: List[str] = config["indicators"]["default"]
        self.data_dir = Path(config["paths"]["data_dir"])
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Optionally include extended indicators
        extended = config["indicators"].get("extended", [])
        if extended:
            self.indicator_names = self.indicator_names + extended

    # -----------------------------------------------------------------
    # Download
    # -----------------------------------------------------------------

    def download_data(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Download OHLCV data from yfinance for all tickers.

        Args:
            start: Start date (default: config train_start).
            end: End date (default: config trade_end).

        Returns:
            DataFrame with columns: date, tic, open, high, low, close,
            volume, adjusted_close.
        """
        start = start or self.start_date
        end = end or self.end_date

        logger.info(
            "Downloading data for %d tickers from %s to %s...",
            len(self.tickers), start, end,
        )

        all_data = []
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start, end=end, auto_adjust=False)

                if hist.empty:
                    logger.warning("No data for %s, skipping.", ticker)
                    continue

                hist = hist.reset_index()

                # Flatten MultiIndex columns if present
                if isinstance(hist.columns, pd.MultiIndex):
                    hist.columns = [
                        col[0] if col[1] == '' or col[1] == ticker
                        else col[0]
                        for col in hist.columns
                    ]

                hist = hist.rename(columns={
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Adj Close": "adjusted_close",
                    "Volume": "volume",
                })

                # Keep only needed columns
                cols_to_keep = [
                    c for c in ["date", "open", "high", "low", "close",
                                "adjusted_close", "volume"]
                    if c in hist.columns
                ]
                hist = hist[cols_to_keep]

                # If no adjusted_close, use close
                if "adjusted_close" not in hist.columns:
                    hist["adjusted_close"] = hist["close"]

                hist["tic"] = ticker
                all_data.append(hist)
                logger.debug("Downloaded %d rows for %s", len(hist), ticker)

            except Exception as e:
                logger.error("Error downloading %s: %s", ticker, e)
                continue

        if not all_data:
            raise RuntimeError("No data downloaded for any ticker.")

        df = pd.concat(all_data, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)

        logger.info(
            "Downloaded %d rows across %d tickers.",
            len(df), df["tic"].nunique(),
        )
        return df

    # -----------------------------------------------------------------
    # Clean
    # -----------------------------------------------------------------

    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw OHLCV data:
            - Drop duplicates
            - Forward-fill then back-fill NaNs per ticker
            - Remove tickers with insufficient data (< 80% of max dates)
            - Ensure all remaining tickers have the same trading days

        Args:
            df: Raw data from download_data().

        Returns:
            Cleaned DataFrame.
        """
        logger.info("Cleaning data...")

        df = df.drop_duplicates(subset=["date", "tic"])

        # Forward/backward fill per ticker
        df = df.sort_values(["tic", "date"])
        numeric_cols = ["open", "high", "low", "close", "adjusted_close", "volume"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df.groupby("tic")[col].transform(
                    lambda x: x.ffill().bfill()
                )

        # Remove tickers with insufficient history (< 80% of max dates)
        all_dates = df["date"].nunique()
        ticker_date_counts = df.groupby("tic")["date"].nunique()
        min_dates = int(all_dates * 0.8)
        valid_tickers = ticker_date_counts[ticker_date_counts >= min_dates].index
        removed_tickers = set(df["tic"].unique()) - set(valid_tickers)
        if removed_tickers:
            logger.warning(
                "Removing tickers with insufficient history (< %d dates): %s",
                min_dates, sorted(removed_tickers),
            )
        df = df[df["tic"].isin(valid_tickers)]

        # Keep only dates where ALL remaining tickers have data
        n_tickers = df["tic"].nunique()
        date_counts = df.groupby("date")["tic"].nunique()
        valid_dates = date_counts[date_counts == n_tickers].index
        df = df[df["date"].isin(valid_dates)]

        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        logger.info(
            "Cleaned data: %d rows, %d tickers, %d dates (range: %s to %s).",
            len(df), df["tic"].nunique(), df["date"].nunique(),
            df["date"].min().strftime("%Y-%m-%d"),
            df["date"].max().strftime("%Y-%m-%d"),
        )
        return df

    # -----------------------------------------------------------------
    # Enrich with indicators & turbulence
    # -----------------------------------------------------------------

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute and add technical indicators specified in config.

        Args:
            df: Cleaned OHLCV data.

        Returns:
            DataFrame with indicator columns appended.
        """
        logger.info(
            "Computing technical indicators: %s",
            ", ".join(self.indicator_names),
        )
        df = add_indicators(df, self.indicator_names, ticker_col="tic")
        return df

    @staticmethod
    def add_turbulence(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute and merge the turbulence index.

        Args:
            df: OHLCV + indicators data.

        Returns:
            DataFrame with 'turbulence' column added.
        """
        logger.info("Computing turbulence index...")
        turb_df = compute_turbulence(df, ticker_col="tic", date_col="date")
        df = df.merge(turb_df, on="date", how="left")
        df["turbulence"] = df["turbulence"].fillna(0.0)
        return df

    # -----------------------------------------------------------------
    # Full pipeline
    # -----------------------------------------------------------------

    def get_processed_data(self, force_download: bool = False) -> pd.DataFrame:
        """
        Execute the full data pipeline: download → clean → enrich.

        Uses cached data if available unless force_download=True.

        Args:
            force_download: If True, re-download from yfinance.

        Returns:
            Fully processed DataFrame ready for environment.
        """
        cache_path = self.data_dir / "processed_data.csv"

        if cache_path.exists() and not force_download:
            logger.info("Loading cached data from %s", cache_path)
            df = pd.read_csv(cache_path, parse_dates=["date"])
            return df

        # Full pipeline
        raw_df = self.download_data()
        clean_df = self.clean_data(raw_df)
        enriched_df = self.add_technical_indicators(clean_df)
        final_df = self.add_turbulence(enriched_df)

        # Cache
        final_df.to_csv(cache_path, index=False)
        logger.info("Saved processed data to %s", cache_path)

        return final_df

    # -----------------------------------------------------------------
    # Update to current date
    # -----------------------------------------------------------------

    def update_data(self) -> pd.DataFrame:
        """
        Update the cached dataset to include data up to today.

        Downloads fresh data and re-runs the full pipeline.

        Returns:
            Updated processed DataFrame.
        """
        from datetime import datetime
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        logger.info("Updating data to %s...", self.end_date)
        return self.get_processed_data(force_download=True)

    # -----------------------------------------------------------------
    # Data splitting
    # -----------------------------------------------------------------

    @staticmethod
    def split_data(
        df: pd.DataFrame,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        Filter DataFrame to a date range [start, end].

        Args:
            df: Full processed DataFrame.
            start: Start date string.
            end: End date string.

        Returns:
            Filtered DataFrame.
        """
        mask = (df["date"] >= start) & (df["date"] <= end)
        return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Benchmark data
# ---------------------------------------------------------------------------

def download_benchmark(
    ticker: str = "^DJI",
    start: str = "2016-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """
    Download benchmark index data for comparison.

    Args:
        ticker: Benchmark ticker symbol.
        start: Start date.
        end: End date.

    Returns:
        DataFrame with columns [date, close].
    """
    logger.info("Downloading benchmark %s from %s to %s", ticker, start, end)
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start, end=end, auto_adjust=True)
    hist = hist.reset_index()

    # Handle MultiIndex columns
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = [col[0] for col in hist.columns]

    hist = hist.rename(columns={"Date": "date", "Close": "close"})
    hist["date"] = pd.to_datetime(hist["date"]).dt.tz_localize(None)
    hist = hist[["date", "close"]].sort_values("date").reset_index(drop=True)
    return hist
