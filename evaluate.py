"""
evaluate.py — Entry Point: Generate Performance Report.

Loads the results from a completed ensemble training run and generates:
    - Metrics table (Cumulative Return, Annual Return, Sharpe, etc.)
    - Cumulative return comparison chart (PNG + interactive HTML)
    - Drawdown chart
    - Model selection timeline

Usage:
    python evaluate.py
    python evaluate.py --config custom_config.yaml
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import load_config, setup_logging
from src.evaluate_metrics import run_evaluation


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DRL Ensemble Trading Strategy Performance"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    logger = setup_logging(config, "drl_ensemble")

    logger.info("=" * 60)
    logger.info("PERFORMANCE EVALUATION")
    logger.info("=" * 60)

    try:
        run_evaluation(config)
    except FileNotFoundError as e:
        logger.error(str(e))
        print(
            "\nError: No results found. Please run 'python train.py' first "
            "to generate ensemble training results."
        )
        sys.exit(1)

    results_dir = config["paths"]["results_dir"]
    print(f"\nAll outputs saved to {results_dir}/:")
    print(f"  • metrics_table.csv          — Performance comparison table")
    print(f"  • cumulative_returns.png     — Static return chart")
    print(f"  • cumulative_returns.html    — Interactive return chart")
    print(f"  • drawdown.png               — Drawdown chart")
    print(f"  • selection_timeline.png     — Model selection by quarter")


if __name__ == "__main__":
    main()
