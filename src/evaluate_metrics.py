"""
evaluate_metrics.py — Performance Evaluation and Visualization Module.

Computes the five metrics from the paper (Section 6.2):
    1. Cumulative Return
    2. Annualized Return
    3. Annualized Volatility
    4. Sharpe Ratio
    5. Max Drawdown

Generates comparative visualizations:
    - Cumulative return curves (Ensemble vs PPO vs A2C vs DDPG vs Buy&Hold)
    - Performance metrics table
    - Drawdown plot
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_loader import download_benchmark

logger = logging.getLogger("drl_ensemble.evaluate")


# ---------------------------------------------------------------------------
# Core Metrics (Section 6.2)
# ---------------------------------------------------------------------------

def compute_cumulative_return(portfolio_values: List[float]) -> float:
    """
    Cumulative Return = (final - initial) / initial

    Section 6.2.1: "subtracting the portfolio's final value from its
    initial value, then dividing by the initial value."
    """
    if len(portfolio_values) < 2:
        return 0.0
    return (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]


def compute_annualized_return(
    portfolio_values: List[float],
    periods_per_year: float = 252.0,
) -> float:
    """
    Annualized Return = geometric average yearly return.

    Section 6.2.1: "geometric average amount of money earned each year."
    """
    if len(portfolio_values) < 2:
        return 0.0
    total_return = portfolio_values[-1] / portfolio_values[0]
    n_periods = len(portfolio_values) - 1
    n_years = n_periods / periods_per_year
    if n_years <= 0:
        return 0.0
    return total_return ** (1 / n_years) - 1


def compute_annualized_volatility(
    portfolio_values: List[float],
    periods_per_year: float = 252.0,
) -> float:
    """
    Annualized Volatility = annualized standard deviation of daily returns.

    Section 6.2.1: "annualized standard deviation of portfolio return."
    """
    if len(portfolio_values) < 3:
        return 0.0
    values = np.array(portfolio_values, dtype=np.float64)
    daily_returns = np.diff(values) / values[:-1]
    return float(np.std(daily_returns, ddof=1) * np.sqrt(periods_per_year))


def compute_sharpe_ratio(
    portfolio_values: List[float],
    risk_free_rate: float = 0.0,
    periods_per_year: float = 252.0,
) -> float:
    """
    Sharpe Ratio (Eq. 19 in the paper):
        Sharpe = (annualized_return - risk_free_rate) / annualized_volatility

    Section 6.2.1: "subtracting the annualized risk free rate from the
    annualized return, then dividing by the annualized volatility."
    """
    ann_return = compute_annualized_return(portfolio_values, periods_per_year)
    ann_vol = compute_annualized_volatility(portfolio_values, periods_per_year)
    if ann_vol < 1e-10:
        return 0.0
    return (ann_return - risk_free_rate) / ann_vol


def compute_max_drawdown(portfolio_values: List[float]) -> float:
    """
    Max Drawdown = maximum percentage loss from peak to trough.

    Section 6.2.1: "maximum percentage loss during the trading period."
    """
    if len(portfolio_values) < 2:
        return 0.0
    values = np.array(portfolio_values, dtype=np.float64)
    peak = np.maximum.accumulate(values)
    drawdown = (values - peak) / peak
    return float(np.min(drawdown))


def compute_all_metrics(
    portfolio_values: List[float],
    label: str = "",
) -> Dict[str, float]:
    """
    Compute all five evaluation metrics.

    Returns:
        Dict with keys: cumulative_return, annualized_return,
        annualized_volatility, sharpe_ratio, max_drawdown.
    """
    metrics = {
        "cumulative_return": compute_cumulative_return(portfolio_values),
        "annualized_return": compute_annualized_return(portfolio_values),
        "annualized_volatility": compute_annualized_volatility(portfolio_values),
        "sharpe_ratio": compute_sharpe_ratio(portfolio_values),
        "max_drawdown": compute_max_drawdown(portfolio_values),
    }

    if label:
        logger.info(
            "  %s: CumRet=%.2f%% AnnRet=%.2f%% AnnVol=%.2f%% "
            "Sharpe=%.4f MaxDD=%.2f%%",
            label,
            metrics["cumulative_return"] * 100,
            metrics["annualized_return"] * 100,
            metrics["annualized_volatility"] * 100,
            metrics["sharpe_ratio"],
            metrics["max_drawdown"] * 100,
        )

    return metrics


# ---------------------------------------------------------------------------
# Results Loading
# ---------------------------------------------------------------------------

def load_ensemble_results(results_dir: str) -> Dict:
    """Load ensemble results from the saved JSON file."""
    results_path = Path(results_dir) / "ensemble_results.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_path}. "
            "Run train.py first to generate results."
        )
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Comparison Table
# ---------------------------------------------------------------------------

def generate_metrics_table(
    results: Dict,
    benchmark_values: Optional[List[float]] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate a comparison table like Table 2 in the paper.

    Args:
        results: Ensemble results dict from ensemble.py.
        benchmark_values: Buy&Hold benchmark portfolio values.
        output_path: Optional CSV path to save.

    Returns:
        DataFrame with metrics for each strategy.
    """
    logger.info("Computing performance metrics...")

    rows = {}

    # Ensemble
    if results.get("ensemble_portfolio_values"):
        rows["Ensemble"] = compute_all_metrics(
            results["ensemble_portfolio_values"], "Ensemble"
        )

    # Individual models
    for algo_name, pv in results.get("individual_results", {}).items():
        if pv:
            rows[algo_name.upper()] = compute_all_metrics(pv, algo_name.upper())

    # Benchmark (Buy & Hold DJIA)
    if benchmark_values:
        rows["DJIA (Buy&Hold)"] = compute_all_metrics(
            benchmark_values, "DJIA"
        )

    # Build DataFrame
    df = pd.DataFrame(rows).T
    df.columns = [
        "Cumulative Return",
        "Annualized Return",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Max Drawdown",
    ]

    # Format percentages
    for col in ["Cumulative Return", "Annualized Return",
                "Annualized Volatility", "Max Drawdown"]:
        df[col] = df[col].apply(lambda x: f"{x * 100:.2f}%")
    df["Sharpe Ratio"] = df["Sharpe Ratio"].apply(lambda x: f"{float(x.replace('%','')):.4f}" if isinstance(x, str) else f"{x:.4f}")

    if output_path:
        df.to_csv(output_path)
        logger.info("Metrics table saved to %s", output_path)

    return df


# ---------------------------------------------------------------------------
# Visualizations — Matplotlib
# ---------------------------------------------------------------------------

def plot_cumulative_returns_matplotlib(
    results: Dict,
    benchmark_values: Optional[List[float]] = None,
    benchmark_dates: Optional[List] = None,
    output_path: str = "results/cumulative_returns.png",
) -> None:
    """
    Plot cumulative return curves like Figure 5 in the paper.

    Shows Ensemble vs PPO vs A2C vs DDPG vs DJIA Buy&Hold.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = {
        "Ensemble": "#FF6B35",
        "PPO": "#2E86AB",
        "A2C": "#A23B72",
        "DDPG": "#1B998B",
        "DJIA (Buy&Hold)": "#666666",
    }

    initial_value = results["ensemble_portfolio_values"][0] if results.get("ensemble_portfolio_values") else 1_000_000

    # Ensemble
    if results.get("ensemble_portfolio_values"):
        pv = results["ensemble_portfolio_values"]
        cum_ret = [(v / initial_value - 1) for v in pv]
        ax.plot(range(len(cum_ret)), cum_ret, color=colors["Ensemble"],
                linewidth=2.5, label="Ensemble", zorder=5)

    # Individual models
    for algo_name, pv in results.get("individual_results", {}).items():
        if pv:
            ind_initial = pv[0] if pv else initial_value
            cum_ret = [(v / ind_initial - 1) for v in pv]
            label = algo_name.upper()
            ax.plot(range(len(cum_ret)), cum_ret,
                    color=colors.get(label, "#999999"),
                    linewidth=1.5, label=label, alpha=0.8)

    # Benchmark
    if benchmark_values:
        bm_initial = benchmark_values[0]
        bm_ret = [(v / bm_initial - 1) for v in benchmark_values]
        ax.plot(range(len(bm_ret)), bm_ret, color=colors["DJIA (Buy&Hold)"],
                linewidth=1.5, linestyle="--", label="DJIA (Buy&Hold)", alpha=0.7)

    ax.set_title(
        "Cumulative Return with Transaction Cost",
        fontsize=16, fontweight="bold",
    )
    ax.set_xlabel("Trading Days", fontsize=12)
    ax.set_ylabel("Cumulative Return", fontsize=12)
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="-")

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Cumulative returns plot saved to %s", output_path)


def plot_drawdown_matplotlib(
    results: Dict,
    output_path: str = "results/drawdown.png",
) -> None:
    """Plot the drawdown curve for the ensemble strategy."""
    if not results.get("ensemble_portfolio_values"):
        return

    pv = np.array(results["ensemble_portfolio_values"], dtype=np.float64)
    peak = np.maximum.accumulate(pv)
    drawdown = (pv - peak) / peak * 100

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(range(len(drawdown)), drawdown, 0,
                    alpha=0.4, color="#E63946")
    ax.plot(range(len(drawdown)), drawdown, color="#E63946", linewidth=1)
    ax.set_title("Ensemble Strategy — Drawdown", fontsize=16, fontweight="bold")
    ax.set_xlabel("Trading Days", fontsize=12)
    ax.set_ylabel("Drawdown (%)", fontsize=12)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Drawdown plot saved to %s", output_path)


# ---------------------------------------------------------------------------
# Visualizations — Plotly (interactive)
# ---------------------------------------------------------------------------

def plot_cumulative_returns_plotly(
    results: Dict,
    benchmark_values: Optional[List[float]] = None,
    output_path: str = "results/cumulative_returns.html",
) -> None:
    """
    Generate an interactive Plotly chart of cumulative returns.

    Saved as HTML for browser viewing.
    """
    fig = go.Figure()

    colors = {
        "Ensemble": "#FF6B35",
        "PPO": "#2E86AB",
        "A2C": "#A23B72",
        "DDPG": "#1B998B",
        "DJIA (Buy&Hold)": "#666666",
    }

    initial_value = results["ensemble_portfolio_values"][0] if results.get("ensemble_portfolio_values") else 1_000_000

    # Ensemble
    if results.get("ensemble_portfolio_values"):
        pv = results["ensemble_portfolio_values"]
        cum_ret = [(v / initial_value - 1) * 100 for v in pv]
        fig.add_trace(go.Scatter(
            y=cum_ret,
            mode="lines",
            name="Ensemble",
            line=dict(color=colors["Ensemble"], width=3),
        ))

    # Individual models
    for algo_name, pv in results.get("individual_results", {}).items():
        if pv:
            ind_initial = pv[0] if pv else initial_value
            cum_ret = [(v / ind_initial - 1) * 100 for v in pv]
            fig.add_trace(go.Scatter(
                y=cum_ret,
                mode="lines",
                name=algo_name.upper(),
                line=dict(color=colors.get(algo_name.upper(), "#999999"), width=1.5),
            ))

    # Benchmark
    if benchmark_values:
        bm_initial = benchmark_values[0]
        bm_ret = [(v / bm_initial - 1) * 100 for v in benchmark_values]
        fig.add_trace(go.Scatter(
            y=bm_ret,
            mode="lines",
            name="DJIA (Buy&Hold)",
            line=dict(color=colors["DJIA (Buy&Hold)"], width=1.5, dash="dash"),
        ))

    fig.update_layout(
        title="Cumulative Return with Transaction Cost (%)",
        xaxis_title="Trading Days",
        yaxis_title="Cumulative Return (%)",
        template="plotly_dark",
        hovermode="x unified",
        font=dict(size=14),
        legend=dict(x=0.01, y=0.99),
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    logger.info("Interactive cumulative returns plot saved to %s", output_path)


# ---------------------------------------------------------------------------
# Selection log visualization
# ---------------------------------------------------------------------------

def plot_selection_timeline(
    selection_log: List[Dict],
    output_path: str = "results/selection_timeline.png",
) -> None:
    """
    Plot which model was selected for each trading quarter.

    Like Table 1 in the paper but as a visual timeline.
    """
    if not selection_log:
        return

    fig, ax = plt.subplots(figsize=(14, 4))

    color_map = {
        "ppo": "#2E86AB",
        "a2c": "#A23B72",
        "ddpg": "#1B998B",
    }

    for i, entry in enumerate(selection_log):
        selected = entry["selected_model"]
        ax.barh(
            0, 1, left=i, height=0.6,
            color=color_map.get(selected, "#999999"),
            edgecolor="white", linewidth=0.5,
        )
        ax.text(
            i + 0.5, 0, selected.upper(),
            ha="center", va="center",
            fontweight="bold", fontsize=8, color="white",
        )

    # Legend
    for algo, color in color_map.items():
        ax.barh(-10, 0, color=color, label=algo.upper())

    ax.set_xlim(-0.1, len(selection_log) + 0.1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("Trading Quarter", fontsize=12)
    ax.set_title("Model Selection Timeline", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Selection timeline saved to %s", output_path)


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def run_evaluation(config: dict) -> None:
    """
    Run the complete evaluation pipeline:
        1. Load ensemble results
        2. Download benchmark data
        3. Compute and display metrics table
        4. Generate all visualizations
    """
    results_dir = config["paths"]["results_dir"]
    results = load_ensemble_results(results_dir)

    # Determine date range from results if available
    trade_start = (
        str(results["ensemble_dates"][0])[:10]
        if results.get("ensemble_dates")
        else config["dates"]["trade_start"]
    )
    trade_end = (
        str(results["ensemble_dates"][-1])[:10]
        if results.get("ensemble_dates")
        else config["dates"]["trade_end"]
    )

    # Download benchmark
    benchmark_df = download_benchmark(
        ticker=config["paper_trading"]["benchmark_ticker"],
        start=trade_start,
        end=trade_end,
    )

    # Normalize benchmark to same initial value
    if results.get("ensemble_portfolio_values"):
        initial = results["ensemble_portfolio_values"][0]
    else:
        initial = config["environment"]["initial_balance"]

    benchmark_values = None
    if not benchmark_df.empty:
        if results.get("ensemble_dates"):
            # Align benchmark dates exactly with ensemble trading days
            benchmark_df["date_str"] = pd.to_datetime(benchmark_df["date"]).dt.strftime("%Y-%m-%d")
            ens_dates_str = [str(d)[:10] for d in results["ensemble_dates"]]
            bench_map = dict(zip(benchmark_df["date_str"], benchmark_df["close"]))
            aligned_closes = []
            last_close = benchmark_df["close"].iloc[0]
            for d in ens_dates_str:
                if d in bench_map:
                    last_close = bench_map[d]
                aligned_closes.append(last_close)
            aligned_s = pd.Series(aligned_closes, dtype=np.float64).ffill().bfill()
            benchmark_values = (aligned_s / aligned_s.iloc[0] * initial).tolist()
        else:
            benchmark_values = (
                benchmark_df["close"] / benchmark_df["close"].iloc[0] * initial
            ).tolist()

    # Metrics table
    metrics_df = generate_metrics_table(
        results,
        benchmark_values=benchmark_values,
        output_path=f"{results_dir}/metrics_table.csv",
    )

    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)
    print(metrics_df.to_string())
    print("=" * 70 + "\n")

    # Visualizations
    plot_cumulative_returns_matplotlib(
        results,
        benchmark_values=benchmark_values,
        output_path=f"{results_dir}/cumulative_returns.png",
    )

    plot_cumulative_returns_plotly(
        results,
        benchmark_values=benchmark_values,
        output_path=f"{results_dir}/cumulative_returns.html",
    )

    plot_drawdown_matplotlib(
        results,
        output_path=f"{results_dir}/drawdown.png",
    )

    plot_selection_timeline(
        results.get("selection_log", []),
        output_path=f"{results_dir}/selection_timeline.png",
    )

    logger.info("Evaluation complete. All results saved to %s/", results_dir)
