"""
paper_trader.py — Paper Trading / Live Simulation Module.

Simulates real-time trading with the trained ensemble model:
    1. Loads the most recent trained model
    2. Downloads current market data
    3. Constructs the current state observation
    4. Predicts the action (shares to buy/sell/hold)
    5. Updates a virtual balance stored in paper_trading/state.json
    6. Logs all trades to paper_trading/trade_log.csv

This enables continuous simulation ("paper trading") without risking
real money, and accumulates new data for future retraining.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.data_loader import DataLoader
from src.environment import StockTradingEnv, create_env
from src.models import load_model, predict_actions
from src.utils import get_device, timestamp_str

logger = logging.getLogger("drl_ensemble.paper_trader")


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

class PaperTradingState:
    """
    Manages the persistent state for paper trading.

    State is saved as JSON with:
        - balance: current cash balance
        - holdings: dict of {ticker: shares}
        - portfolio_history: list of {date, portfolio_value}
        - trade_history: list of {date, ticker, action, shares, price}
        - last_model: algorithm name of the model in use
        - last_update: timestamp of last update
    """

    def __init__(self, state_dir: str, config: dict):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_dir / "state.json"
        self.trade_log_path = self.state_dir / "trade_log.csv"
        self.config = config

        if self.state_path.exists():
            self.state = self._load_state()
        else:
            self.state = self._init_state()

    def _init_state(self) -> Dict:
        """Initialize a fresh paper trading state."""
        initial_balance = self.config["paper_trading"]["initial_balance"]
        tickers = self.config["tickers"]

        state = {
            "balance": initial_balance,
            "holdings": {tic: 0 for tic in tickers},
            "portfolio_history": [{
                "date": datetime.now().strftime("%Y-%m-%d"),
                "portfolio_value": initial_balance,
                "balance": initial_balance,
            }],
            "trade_history": [],
            "last_model": None,
            "last_update": None,
            "initial_balance": initial_balance,
        }

        self._save_state(state)
        logger.info(
            "Initialized paper trading state with $%s balance.",
            f"{initial_balance:,.2f}",
        )
        return state

    def _load_state(self) -> Dict:
        """Load state from JSON file."""
        with open(self.state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        if "holdings" in state and isinstance(state["holdings"], dict):
            state["holdings"] = {k: int(v) for k, v in state["holdings"].items()}
        if "balance" in state:
            state["balance"] = float(state["balance"])
        logger.info("Loaded paper trading state from %s", self.state_path)
        return state

    def _save_state(self, state: Optional[Dict] = None) -> None:
        """Save state to JSON file with native Python types."""
        state = state or self.state

        def _to_native(obj):
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: _to_native(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_native(v) for v in obj]
            return obj

        serializable = _to_native(state)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

    def update(
        self,
        balance: float,
        holdings: Dict[str, int],
        prices: Dict[str, float],
        trades: List[Dict],
        model_name: str,
    ) -> None:
        """
        Update the paper trading state after a trading step.

        Args:
            balance: New cash balance.
            holdings: New holdings {ticker: shares}.
            prices: Current prices {ticker: price}.
            trades: List of trades executed.
            model_name: Name of the model used.
        """
        self.state["balance"] = balance
        self.state["holdings"] = holdings
        self.state["last_model"] = model_name
        self.state["last_update"] = datetime.now().isoformat()

        # Portfolio value
        portfolio_value = balance + sum(
            prices.get(tic, 0) * shares
            for tic, shares in holdings.items()
        )

        self.state["portfolio_history"].append({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "portfolio_value": portfolio_value,
            "balance": balance,
        })

        # Add trades
        self.state["trade_history"].extend(trades)

        self._save_state()

        # Append to CSV trade log
        if trades:
            trades_df = pd.DataFrame(trades)
            header = not self.trade_log_path.exists()
            trades_df.to_csv(
                self.trade_log_path,
                mode="a",
                header=header,
                index=False,
            )

        logger.info(
            "Updated state: Balance=$%s, Portfolio=$%s, Trades=%d",
            f"{balance:,.2f}", f"{portfolio_value:,.2f}", len(trades),
        )

    def get_summary(self) -> str:
        """Return a human-readable summary of the current state."""
        balance = self.state["balance"]
        holdings = self.state["holdings"]
        hist = self.state["portfolio_history"]
        initial = self.state["initial_balance"]

        current_value = hist[-1]["portfolio_value"] if hist else balance
        pnl = current_value - initial
        pnl_pct = (pnl / initial) * 100

        active_positions = {
            k: v for k, v in holdings.items() if v > 0
        }

        lines = [
            "=" * 50,
            "PAPER TRADING STATUS",
            "=" * 50,
            f"Balance:           ${balance:>12,.2f}",
            f"Portfolio Value:    ${current_value:>12,.2f}",
            f"P&L:               ${pnl:>12,.2f} ({pnl_pct:+.2f}%)",
            f"Active Positions:  {len(active_positions)}",
            f"Total Trades:      {len(self.state['trade_history'])}",
            f"Last Model:        {self.state.get('last_model', 'N/A')}",
            f"Last Update:       {self.state.get('last_update', 'N/A')}",
        ]

        if active_positions:
            lines.append("-" * 50)
            lines.append("Holdings:")
            for tic, shares in sorted(active_positions.items()):
                lines.append(f"  {tic:<8} {shares:>6} shares")

        lines.append("=" * 50)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Paper Trading Engine
# ---------------------------------------------------------------------------

class PaperTrader:
    """
    Simulates a trading step using the trained ensemble model.

    On each run:
        1. Determines which model to use (latest trained)
        2. Downloads current market data
        3. Builds the state vector
        4. Predicts actions
        5. Simulates execution (with transaction costs)
        6. Updates persistent state
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = get_device(config)
        self.tickers = config["tickers"]
        self.tx_cost = config["environment"]["transaction_cost_pct"]
        self.hmax = config["environment"]["hmax"]
        self.models_dir = Path(config["paths"]["models_dir"])
        self.pt_state = PaperTradingState(
            config["paths"]["paper_trading_dir"], config
        )

    def find_latest_model(self) -> tuple[str, Path]:
        """
        Find the most recently saved model.

        Looks for the latest selection log to determine which algorithm
        was selected, then loads that model.

        Returns:
            (algorithm_name, model_path)
        """
        results_dir = Path(self.config["paths"]["results_dir"])
        selection_log_path = results_dir / "selection_log.json"

        if selection_log_path.exists():
            with open(selection_log_path, "r") as f:
                selection_log = json.load(f)
            if selection_log:
                last_entry = selection_log[-1]
                algo_name = last_entry["selected_model"]
                window_idx = last_entry["window"]

                model_path = self.models_dir / f"{algo_name}_window_{window_idx}.zip"
                if model_path.exists():
                    return algo_name, model_path

        # Fallback: find any available model
        for algo_name in ["ppo", "a2c", "ddpg"]:
            model_files = sorted(
                self.models_dir.glob(f"{algo_name}_*.zip"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if model_files:
                return algo_name, model_files[0]

        raise FileNotFoundError(
            "No trained models found. Run train.py first."
        )

    def get_current_data(self, lookback_days: int = 60) -> pd.DataFrame:
        """
        Download recent market data for building the current state.

        Args:
            lookback_days: Number of days of history to download
                          (needed for indicator warm-up).

        Returns:
            Processed DataFrame with indicators.
        """
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (
            datetime.now() - timedelta(days=lookback_days)
        ).strftime("%Y-%m-%d")

        loader = DataLoader(self.config)
        raw_df = loader.download_data(start=start_date, end=end_date)
        clean_df = loader.clean_data(raw_df)
        enriched_df = loader.add_technical_indicators(clean_df)
        final_df = loader.add_turbulence(enriched_df)

        return final_df

    def execute_step(self) -> Dict:
        """
        Execute a single paper trading step.

        Returns:
            Dictionary with step results.
        """
        logger.info("=" * 50)
        logger.info("PAPER TRADING STEP — %s", datetime.now().strftime("%Y-%m-%d %H:%M"))
        logger.info("=" * 50)

        # 1. Find and load the best model
        algo_name, model_path = self.find_latest_model()
        logger.info("Using model: %s (%s)", algo_name.upper(), model_path)

        # 2. Download current data
        current_data = self.get_current_data()

        if current_data.empty:
            logger.warning("No current data available. Market may be closed.")
            return {"status": "no_data"}

        # 3. Create a mini-environment with the latest data
        env = create_env(current_data, self.config, mode="trade")
        env_tickers = env.tickers

        # Set the environment to the current paper trading state
        state_data = self.pt_state.state
        env.balance = float(state_data["balance"])
        for i, tic in enumerate(env_tickers):
            if tic in state_data["holdings"]:
                env.holdings[i] = float(state_data["holdings"][tic])

        # 4. Load model and predict
        model = load_model(algo_name, str(model_path), device=self.device)

        # Get the latest state from the environment (last day)
        env.day = env.terminal_day - 1
        env._data = env._get_day_data(env.day)
        state = env._get_state()

        action = predict_actions(model, state, deterministic=True)
        actions_int = (np.clip(action, -1.0, 1.0) * self.hmax).astype(int)

        # 5. Simulate execution
        prices = env._get_prices()
        balance = float(state_data["balance"])
        holdings = {k: int(v) for k, v in state_data["holdings"].items()}
        trades_executed = []

        # Sell first
        for i, tic in enumerate(env_tickers):
            if actions_int[i] < 0:
                current_shares = int(holdings.get(tic, 0))
                shares_to_sell = min(int(abs(actions_int[i])), current_shares)
                if shares_to_sell > 0:
                    sell_value = float(prices[i]) * shares_to_sell
                    tx_cost = sell_value * self.tx_cost
                    balance += sell_value - tx_cost
                    holdings[tic] = current_shares - shares_to_sell
                    trades_executed.append({
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "ticker": tic,
                        "action": "SELL",
                        "shares": int(shares_to_sell),
                        "price": float(prices[i]),
                        "value": float(sell_value),
                        "cost": float(tx_cost),
                    })

        # Then buy
        for i, tic in enumerate(env_tickers):
            if actions_int[i] > 0:
                cost_per_share = float(prices[i]) * (1 + self.tx_cost)
                if cost_per_share <= 0:
                    continue
                max_affordable = int(balance // cost_per_share)
                shares_to_buy = min(int(actions_int[i]), max_affordable)
                if shares_to_buy > 0:
                    buy_value = float(prices[i]) * shares_to_buy
                    tx_cost = buy_value * self.tx_cost
                    balance -= (buy_value + tx_cost)
                    holdings[tic] = int(holdings.get(tic, 0)) + shares_to_buy
                    trades_executed.append({
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "ticker": tic,
                        "action": "BUY",
                        "shares": int(shares_to_buy),
                        "price": float(prices[i]),
                        "value": float(buy_value),
                        "cost": float(tx_cost),
                    })

        # 6. Update state
        current_prices = {
            tic: float(prices[i]) for i, tic in enumerate(env_tickers)
        }
        self.pt_state.update(
            balance=float(balance),
            holdings=holdings,
            prices=current_prices,
            trades=trades_executed,
            model_name=algo_name,
        )

        # 7. Print summary
        summary = self.pt_state.get_summary()
        print(summary)
        logger.info("\n%s", summary)

        if trades_executed:
            print("\nTrades Executed:")
            for t in trades_executed:
                print(
                    f"  {t['action']:<4} {t['shares']:>4} x {t['ticker']:<6} "
                    f"@ ${t['price']:.2f} = ${t['value']:,.2f} "
                    f"(cost: ${t['cost']:.2f})"
                )
        else:
            print("\nNo trades executed (HOLD).")

        return {
            "status": "success",
            "trades": trades_executed,
            "balance": float(balance),
            "model": algo_name,
        }


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def run_paper_trading(config: dict) -> None:
    """Entry point for paper trading from the CLI script."""
    trader = PaperTrader(config)
    trader.execute_step()
