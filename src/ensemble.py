"""
ensemble.py — Ensemble Strategy with Rolling Windows.

Implements the ensemble strategy from Section 5.4 of Yang et al. (2020):

    Step 1: Growing window of n months → retrain PPO, A2C, DDPG
    Step 2: Validate with 3-month rolling window → pick best Sharpe ratio
    Step 3: Use the best agent to trade the next quarter

The Sharpe ratio for selection (Section 5.4, Eq. 19):
    Sharpe = (r̄_p - r_f) / σ_p

Where r̄_p is expected portfolio return, r_f is risk-free rate,
and σ_p is portfolio standard deviation.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data_loader import DataLoader
from src.environment import StockTradingEnv, create_env
from src.models import (
    create_model,
    get_total_timesteps,
    load_model,
    predict_actions,
    save_model,
    train_model,
)
from src.turbulence import calculate_turbulence_threshold
from src.utils import get_device, set_seed

logger = logging.getLogger("drl_ensemble.ensemble")

# Algorithms to ensemble
ALGO_NAMES = ["ppo", "a2c", "ddpg"]


def compute_sharpe_ratio(
    portfolio_values: List[float],
    risk_free_rate: float = 0.0,
    periods_per_year: float = 252.0,
) -> float:
    """
    Compute the annualized Sharpe ratio from a sequence of portfolio values.

    Sharpe = (annualized_return - risk_free_rate) / annualized_volatility

    Args:
        portfolio_values: List of portfolio values over time.
        risk_free_rate: Annual risk-free rate (default 0).
        periods_per_year: Trading periods per year (252 for daily).

    Returns:
        Sharpe ratio (returns 0.0 if insufficient data or zero volatility).
    """
    if len(portfolio_values) < 2:
        return 0.0

    values = np.array(portfolio_values, dtype=np.float64)
    daily_returns = np.diff(values) / values[:-1]

    if len(daily_returns) == 0:
        return 0.0

    mean_return = np.mean(daily_returns)
    std_return = np.std(daily_returns, ddof=1) if len(daily_returns) > 1 else 1e-8

    if std_return < 1e-10:
        return 0.0

    # Annualize
    annualized_return = mean_return * periods_per_year
    annualized_std = std_return * np.sqrt(periods_per_year)

    sharpe = (annualized_return - risk_free_rate) / annualized_std
    return float(sharpe)


def generate_rolling_windows(
    trade_start: str,
    trade_end: str,
    rebalance_months: int = 3,
) -> List[Tuple[str, str]]:
    """
    Generate quarterly (or custom) trading windows for the ensemble.

    Args:
        trade_start: Start of trading period.
        trade_end: End of trading period.
        rebalance_months: Months per window.

    Returns:
        List of (window_start, window_end) date strings.
    """
    start = pd.Timestamp(trade_start)
    end = pd.Timestamp(trade_end)

    windows = []
    current = start

    while current < end:
        window_end = current + pd.DateOffset(months=rebalance_months) - pd.DateOffset(days=1)
        if window_end > end:
            window_end = end
        windows.append((
            current.strftime("%Y-%m-%d"),
            window_end.strftime("%Y-%m-%d"),
        ))
        current = current + pd.DateOffset(months=rebalance_months)

    return windows


class EnsembleStrategy:
    """
    Implements the Ensemble Strategy as described in Section 5.4.

    For each trading quarter:
        1. Train PPO, A2C, DDPG on growing training window
        2. Validate on validation window → select best by Sharpe
        3. Trade with selected model for the quarter
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = get_device(config)
        self.seed = config["training"]["seed"]

        # Dates
        self.train_start = config["dates"]["train_start"]
        self.validation_months = config["ensemble"]["validation_window_months"]
        self.rebalance_months = config["ensemble"]["rebalance_window_months"]
        self.trade_start = config["dates"]["trade_start"]
        self.trade_end = config["dates"]["trade_end"]

        # Paths
        self.models_dir = Path(config["paths"]["models_dir"])
        self.results_dir = Path(config["paths"]["results_dir"])
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Results tracking
        self.ensemble_portfolio_values: List[float] = []
        self.ensemble_dates: List = []
        self.selection_log: List[Dict] = []
        self.individual_results: Dict[str, List[float]] = {
            algo: [] for algo in ALGO_NAMES
        }

    def run(self, full_df: pd.DataFrame) -> Dict:
        """
        Execute the full ensemble strategy.

        Args:
            full_df: Complete processed data from DataLoader.

        Returns:
            Dictionary with results:
                - ensemble_portfolio_values
                - ensemble_dates
                - selection_log
                - individual_results (per-algo portfolio values)
        """
        set_seed(self.seed)

        # Generate trading windows
        trade_windows = generate_rolling_windows(
            self.trade_start, self.trade_end, self.rebalance_months
        )

        logger.info(
            "Ensemble Strategy: %d trading windows from %s to %s",
            len(trade_windows), self.trade_start, self.trade_end,
        )

        # Calculate turbulence threshold from training data
        train_data = DataLoader.split_data(full_df, self.train_start, self.trade_start)
        turb_threshold = self.config["environment"].get("turbulence_threshold")
        if turb_threshold is None and "turbulence" in train_data.columns:
            turb_threshold = calculate_turbulence_threshold(
                train_data["turbulence"], percentile=90.0
            )
            # If threshold is inf (no valid turbulence data in training set),
            # recalculate from the full dataset instead
            if turb_threshold == float("inf") and "turbulence" in full_df.columns:
                turb_threshold = calculate_turbulence_threshold(
                    full_df["turbulence"], percentile=90.0
                )
            logger.info("Turbulence threshold: %.2f", turb_threshold)

        # Track cumulative portfolio across windows
        last_portfolio_value = self.config["environment"]["initial_balance"]
        ind_last_portfolio_values = {
            algo: self.config["environment"]["initial_balance"] for algo in ALGO_NAMES
        }

        for window_idx, (win_start, win_end) in enumerate(trade_windows):
            logger.info(
                "=" * 60 + "\n"
                "Window %d/%d: Trading %s to %s\n" + "=" * 60,
                window_idx + 1, len(trade_windows), win_start, win_end,
            )

            # ----- Training window: from train_start to validation_start -----
            # Validation window ends at win_start (exclusive)
            val_end = pd.Timestamp(win_start) - pd.DateOffset(days=1)
            val_start = val_end - pd.DateOffset(months=self.validation_months) + pd.DateOffset(days=1)
            train_end = val_start - pd.DateOffset(days=1)

            val_start_str = val_start.strftime("%Y-%m-%d")
            val_end_str = val_end.strftime("%Y-%m-%d")
            train_end_str = train_end.strftime("%Y-%m-%d")

            logger.info(
                "  Training:   %s to %s", self.train_start, train_end_str
            )
            logger.info(
                "  Validation: %s to %s", val_start_str, val_end_str
            )

            # Get data subsets
            train_df = DataLoader.split_data(full_df, self.train_start, train_end_str)
            val_df = DataLoader.split_data(full_df, val_start_str, val_end_str)
            trade_df = DataLoader.split_data(full_df, win_start, win_end)

            if train_df.empty or val_df.empty or trade_df.empty:
                logger.warning(
                    "  Skipping window %d: insufficient data.", window_idx + 1
                )
                continue

            # ----- Train all three models -----
            sharpe_scores = {}
            trained_models = {}

            for algo_name in ALGO_NAMES:
                logger.info("  Training %s...", algo_name.upper())

                try:
                    # Create training environment
                    train_env = create_env(train_df, self.config, mode="train")
                    train_env.turbulence_threshold = turb_threshold

                    # Create and train model
                    model = create_model(
                        algo_name, train_env, self.config, device=self.device
                    )
                    timesteps = get_total_timesteps(self.config, algo_name)
                    model = train_model(model, timesteps, algo_name)

                    # Save model
                    model_path = (
                        self.models_dir
                        / f"{algo_name}_window_{window_idx}"
                    )
                    save_model(model, str(model_path))
                    trained_models[algo_name] = model

                    # ----- Validate: compute Sharpe on validation data -----
                    val_env = create_env(val_df, self.config, mode="trade")
                    val_env.turbulence_threshold = turb_threshold
                    val_env.initial_balance = last_portfolio_value
                    val_env.balance = last_portfolio_value

                    val_portfolio = self._run_model_on_env(model, val_env)
                    sharpe = compute_sharpe_ratio(val_portfolio)
                    sharpe_scores[algo_name] = sharpe

                    logger.info(
                        "  %s validation Sharpe: %.4f",
                        algo_name.upper(), sharpe,
                    )

                except Exception as e:
                    logger.error(
                        "  Error training/validating %s: %s",
                        algo_name.upper(), e,
                    )
                    sharpe_scores[algo_name] = float("-inf")

            # ----- Select best model by Sharpe (Section 5.4, Step 2) -----
            best_algo = max(sharpe_scores, key=sharpe_scores.get)
            logger.info(
                "  Selected: %s (Sharpe=%.4f)",
                best_algo.upper(), sharpe_scores[best_algo],
            )

            self.selection_log.append({
                "window": window_idx,
                "trade_start": win_start,
                "trade_end": win_end,
                "sharpe_scores": {k: round(v, 4) for k, v in sharpe_scores.items()},
                "selected_model": best_algo,
            })

            # ----- Trade with selected model (Section 5.4, Step 3) -----
            if best_algo in trained_models:
                trade_env = create_env(trade_df, self.config, mode="trade")
                trade_env.turbulence_threshold = turb_threshold
                trade_env.initial_balance = last_portfolio_value
                trade_env.balance = last_portfolio_value

                ensemble_pv = self._run_model_on_env(
                    trained_models[best_algo], trade_env
                )

                # Append to cumulative results (skip first value to avoid
                # double-counting the starting portfolio value)
                trade_dates = trade_env.get_dates()
                if self.ensemble_portfolio_values:
                    self.ensemble_portfolio_values.extend(ensemble_pv[1:])
                    self.ensemble_dates.extend(trade_dates[1:])
                else:
                    self.ensemble_portfolio_values.extend(ensemble_pv)
                    self.ensemble_dates.extend(trade_dates)

                last_portfolio_value = ensemble_pv[-1]

                logger.info(
                    "  Window portfolio: $%s → $%s",
                    f"{ensemble_pv[0]:,.2f}",
                    f"{ensemble_pv[-1]:,.2f}",
                )

            # ----- Also run individual models for comparison -----
            for algo_name in ALGO_NAMES:
                if algo_name in trained_models:
                    try:
                        ind_env = create_env(trade_df, self.config, mode="trade")
                        ind_env.turbulence_threshold = turb_threshold
                        ind_env.initial_balance = ind_last_portfolio_values[algo_name]
                        ind_env.balance = ind_last_portfolio_values[algo_name]
                        ind_pv = self._run_model_on_env(
                            trained_models[algo_name], ind_env
                        )
                        self.individual_results[algo_name].extend(
                            ind_pv[1:] if self.individual_results[algo_name]
                            else ind_pv
                        )
                        ind_last_portfolio_values[algo_name] = ind_pv[-1]
                    except Exception as e:
                        logger.error(
                            "  Error running %s individually: %s",
                            algo_name.upper(), e,
                        )

        # ----- Save results -----
        results = {
            "ensemble_portfolio_values": self.ensemble_portfolio_values,
            "ensemble_dates": [str(d) for d in self.ensemble_dates],
            "selection_log": self.selection_log,
            "individual_results": {
                k: v for k, v in self.individual_results.items()
            },
        }

        self._save_results(results)
        return results

    def _run_model_on_env(
        self,
        model,
        env: StockTradingEnv,
    ) -> List[float]:
        """
        Run a trained model through an environment episode.

        Args:
            model: Trained SB3 model.
            env: Trading environment.

        Returns:
            List of portfolio values at each step.
        """
        state, _ = env.reset()
        done = False

        while not done:
            action = predict_actions(model, state, deterministic=True)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        return env.get_portfolio_values()

    def _save_results(self, results: Dict) -> None:
        """Save results to JSON for later evaluation."""
        results_path = self.results_dir / "ensemble_results.json"

        # Convert numpy types to Python native types
        serializable = {}
        for key, value in results.items():
            if isinstance(value, list):
                serializable[key] = [
                    float(v) if isinstance(v, (np.floating, float)) else v
                    for v in value
                ]
            elif isinstance(value, dict):
                serializable[key] = {}
                for k2, v2 in value.items():
                    if isinstance(v2, list):
                        serializable[key][k2] = [
                            float(x) if isinstance(x, (np.floating, float)) else x
                            for x in v2
                        ]
                    else:
                        serializable[key][k2] = v2
            else:
                serializable[key] = value

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, default=str)

        logger.info("Results saved to %s", results_path)

        # Also save selection log as readable table
        log_path = self.results_dir / "selection_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.selection_log, f, indent=2)

        logger.info("Selection log saved to %s", log_path)
