"""
paper_trade.py — Entry Point: Paper Trading Simulation.

Runs a single paper trading step using the latest trained model:
    1. Loads the most recently selected ensemble model
    2. Downloads current market data
    3. Predicts actions for today
    4. Simulates execution with transaction costs
    5. Updates the virtual portfolio balance in paper_trading/state.json

Usage:
    python paper_trade.py                    # Run one step
    python paper_trade.py --status           # Show current portfolio status
    python paper_trade.py --reset            # Reset paper trading balance
    python paper_trade.py --config alt.yaml  # Use alternate config
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import load_config, setup_logging
from src.paper_trader import PaperTrader, PaperTradingState, run_paper_trading


def main():
    parser = argparse.ArgumentParser(
        description="Paper Trading Simulation with DRL Ensemble"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current paper trading portfolio status",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset paper trading to initial balance",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    logger = setup_logging(config, "drl_ensemble")

    if args.status:
        # Just show status
        pt_state = PaperTradingState(
            config["paths"]["paper_trading_dir"], config
        )
        print(pt_state.get_summary())
        return

    if args.reset:
        # Reset state
        pt_state = PaperTradingState(
            config["paths"]["paper_trading_dir"], config
        )
        pt_state.state = pt_state._init_state()
        print("Paper trading state reset to initial balance.")
        return

    # Execute paper trading step
    logger.info("=" * 60)
    logger.info("PAPER TRADING STEP")
    logger.info("=" * 60)

    try:
        run_paper_trading(config)
    except FileNotFoundError as e:
        logger.error(str(e))
        print(
            "\nError: No trained models found. "
            "Please run 'python train.py' first."
        )
        sys.exit(1)
    except Exception as e:
        logger.error("Paper trading error: %s", e, exc_info=True)
        print(f"\nError during paper trading: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
