"""
train.py — Entry Point: Train the DRL Ensemble Strategy.

Executes the full ensemble pipeline:
    1. Load configuration from config.yaml
    2. Download and preprocess market data
    3. Run the ensemble strategy (rolling window training + Sharpe selection)
    4. Save trained models and results

Usage:
    python train.py
    python train.py --config custom_config.yaml
    python train.py --update-data    # Force fresh data download
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import load_config, setup_logging, set_seed, ensure_dirs, get_device
from src.data_loader import DataLoader
from src.ensemble import EnsembleStrategy


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Train the DRL Ensemble Trading Strategy"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--update-data",
        action="store_true",
        help="Force re-download of market data from yfinance",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup
    ensure_dirs(config)
    logger = setup_logging(config, "drl_ensemble")
    set_seed(config["training"]["seed"])

    device = get_device(config)
    logger.info("Using device: %s", device)

    # ---- Step 1: Data Pipeline ----
    logger.info("=" * 60)
    logger.info("STEP 1: Data Pipeline")
    logger.info("=" * 60)

    data_loader = DataLoader(config)
    if args.update_data:
        full_df = data_loader.update_data()
    else:
        full_df = data_loader.get_processed_data()

    logger.info(
        "Data loaded: %d rows, %d tickers, date range %s to %s",
        len(full_df),
        full_df["tic"].nunique(),
        full_df["date"].min().strftime("%Y-%m-%d"),
        full_df["date"].max().strftime("%Y-%m-%d"),
    )

    # ---- Step 2: Ensemble Training ----
    logger.info("=" * 60)
    logger.info("STEP 2: Ensemble Strategy Training")
    logger.info("=" * 60)

    start_time = time.time()
    ensemble = EnsembleStrategy(config)
    results = ensemble.run(full_df)
    elapsed = time.time() - start_time

    # ---- Summary ----
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info("Elapsed time: %.1f minutes", elapsed / 60)
    logger.info("Trading windows processed: %d", len(results.get("selection_log", [])))

    if results.get("ensemble_portfolio_values"):
        initial = results["ensemble_portfolio_values"][0]
        final = results["ensemble_portfolio_values"][-1]
        cum_return = (final - initial) / initial * 100
        logger.info("Ensemble cumulative return: %.2f%%", cum_return)
        logger.info("Final portfolio value: $%s", f"{final:,.2f}")

    logger.info("Models saved to: %s/", config["paths"]["models_dir"])
    logger.info("Results saved to: %s/", config["paths"]["results_dir"])
    logger.info("\nRun 'python evaluate.py' to generate performance report.")


if __name__ == "__main__":
    main()
