# DRL Ensemble Trading System

Deep Reinforcement Learning ensemble strategy for automated stock trading, implementing the methodologies from:

1. **Yang et al. (2020)** — *"Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy"* (ICAIF'20)
2. **Liu et al. (2021)** — *"FinRL: Deep Reinforcement Learning Framework to Automate Trading in Quantitative Finance"* (ICAIF'21)

## Overview

The system trains three actor-critic DRL agents — **PPO**, **A2C**, and **DDPG** — on historical market data and uses a **rolling-window ensemble strategy** to dynamically select the best-performing agent each quarter based on **Sharpe Ratio**.

### Key Features

- **Exact MDP implementation** from the papers (181-dim state space, continuous action space)
- **Turbulence index** for crash protection (automatic liquidation during extreme market events)
- **Rolling-window ensemble** with Sharpe-based model selection
- **Configurable** via `config.yaml` (tickers, dates, hyperparameters, costs)
- **Paper trading simulation** with persistent virtual balance
- **Comparative visualizations** (Ensemble vs individual agents vs DJIA Buy&Hold)
- **GPU/CPU auto-detection** (CUDA support)

---

## Project Structure

```
drl_ensemble_trading/
├── config.yaml                 # Central configuration (tickers, dates, hyperparams)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── train.py                    # Entry point: ensemble training
├── evaluate.py                 # Entry point: metrics & charts
├── paper_trade.py              # Entry point: live simulation
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data download, cleaning, caching
│   ├── environment.py          # Gymnasium MDP environment
│   ├── models.py               # SB3 model management (PPO/A2C/DDPG)
│   ├── ensemble.py             # Rolling window + Sharpe selection
│   ├── evaluate_metrics.py     # Performance metrics & visualizations
│   ├── paper_trader.py         # Paper trading simulation engine
│   ├── turbulence.py           # Financial turbulence index
│   ├── indicators.py           # Extensible technical indicators
│   └── utils.py                # Config, logging, device, seeds
│
├── data/                       # Downloaded & cached market data
├── models/                     # Saved model weights (.zip)
├── results/                    # Charts, metrics, reports
├── logs/                       # Training & execution logs
└── paper_trading/              # Virtual portfolio state
```

---

## Setup & Installation

### 1. Prerequisites

- **Python 3.10+** (tested on 3.10, 3.11)
- **pip** (package manager)
- **GPU** (optional): NVIDIA GPU with CUDA for faster training

### 2. Create a Virtual Environment

```bash
cd drl_ensemble_trading

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note for GPU users**: If pip installs CPU-only PyTorch, install the CUDA version manually:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

### 4. Verify Installation

```bash
python -c "import stable_baselines3; import torch; print(f'SB3: {stable_baselines3.__version__}'); print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Configuration

All parameters are in `config.yaml`. Key settings:

| Parameter | Default | Description |
|---|---|---|
| `tickers` | DJIA 30 | Stock universe |
| `dates.train_start` | 2009-01-01 | Training period start |
| `dates.trade_end` | 2024-12-31 | Trading period end (or "today") |
| `ensemble.rebalance_window_months` | 3 | Retrain frequency |
| `environment.initial_balance` | 1,000,000 | Starting capital |
| `environment.transaction_cost_pct` | 0.001 | 0.1% per trade |
| `environment.hmax` | 100 | Max shares per buy action |
| `training.device` | auto | "auto", "cuda", or "cpu" |

---

## Usage

### Quick Start (Demo / Fast Verification)

To verify the full pipeline end-to-end in ~2 minutes (downloads 2022-2023 data, trains 2 quarterly windows with PPO, A2C, and DDPG, selects best by Sharpe, and generates reports):

```bash
# 1. Fast training run
python train.py --config config_quick.yaml

# 2. Generate evaluation table & charts
python evaluate.py --config config_quick.yaml

# 3. Simulate a live paper trade
python paper_trade.py --config config_quick.yaml
```

---

### Step 1: Train the Full Ensemble Strategy (Paper Settings)

```bash
# Standard run using config.yaml (full historical data from 2009)
python train.py
```

This will:
1. Download historical data from yfinance (cached in `data/`)
2. Compute technical indicators (MACD, RSI, CCI, ADX) and turbulence index
3. For each quarterly trading window:
   - Train PPO, A2C, and DDPG on the growing training window
   - Validate each on a 3-month validation window
   - Select the best model by Sharpe Ratio
   - Run the selected model for the trading quarter
4. Save all models to `models/` and results to `results/`

**Options:**

```bash
# Force re-download data from yfinance
python train.py --update-data

# Use a custom config file
python train.py --config my_config.yaml
```

> **Training Time**: ~30-60 minutes on CPU, ~15-30 minutes with GPU, depending on the number of trading windows.

### Step 2: Generate Performance Report

```bash
python evaluate.py
```

**Outputs** (saved to `results/`):

| File | Description |
|---|---|
| `metrics_table.csv` | Performance comparison table |
| `cumulative_returns.png` | Static cumulative return chart |
| `cumulative_returns.html` | Interactive Plotly chart (open in browser) |
| `drawdown.png` | Drawdown visualization |
| `selection_timeline.png` | Which model was selected each quarter |

### Step 3: Paper Trading (Live Simulation)

```bash
# Run a trading step with current market data
python paper_trade.py

# Check current portfolio status
python paper_trade.py --status

# Reset virtual balance to initial
python paper_trade.py --reset
```

Paper trading state is persisted in `paper_trading/state.json`. Run this daily after market close to simulate live trading.

---

## Technical Details

### MDP Formulation (from the paper)

| Component | Definition |
|---|---|
| **State** | `s = [balance, prices(30), holdings(30), MACD(30), RSI(30), CCI(30), ADX(30)]` = 181 dims |
| **Action** | `a ∈ [-1, 1]^30` continuous, scaled to `[-h_max, h_max]` integer shares |
| **Reward** | `r = portfolio_value(t+1) - portfolio_value(t)` |
| **Portfolio Value** | `v = p^T · h + b` (prices × holdings + cash) |
| **Transaction Cost** | 0.1% of each trade value |

### Turbulence Index

```
turbulence_t = (y_t - μ) · Σ^{-1} · (y_t - μ)^T
```

When `turbulence_t > threshold` (auto-calculated as 90th percentile):
- All shares are sold (liquidation)
- No buying until turbulence drops below threshold

### Ensemble Selection (Rolling Windows)

```
For each quarter in trading period:
  1. Train PPO, A2C, DDPG on growing window
  2. Validate: compute Sharpe = (r̄_p - r_f) / σ_p
  3. Select model with highest Sharpe
  4. Trade with selected model for the quarter
```

---

## Extending the System

### Adding a New Technical Indicator

Edit `src/indicators.py`:

```python
from src.indicators import register_indicator

@register_indicator("my_indicator")
def compute_my_indicator(df: pd.DataFrame) -> pd.Series:
    # Your computation using df["close"], df["high"], etc.
    return result_series
```

Then add `"my_indicator"` to the `indicators.default` list in `config.yaml`.

### Adding a New DRL Algorithm

1. Install the algorithm (e.g., from `sb3-contrib`)
2. Add it to `ALGORITHM_MAP` in `src/models.py`
3. Add hyperparameters to `config.yaml` under `models`
4. Add the algorithm name to `ALGO_NAMES` in `src/ensemble.py`

---

## References

1. Yang, H., Liu, X.-Y., Zhong, S., & Walid, A. (2020). *Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy*. ICAIF'20. https://doi.org/10.1145/3383455.3422540
2. Liu, X.-Y., Yang, H., Gao, J., & Wang, C. D. (2021). *FinRL: Deep Reinforcement Learning Framework to Automate Trading in Quantitative Finance*. ICAIF'21. https://doi.org/10.1145/3490354.3494366
3. Kritzman, M. & Li, Y. (2010). *Skulls, Financial Turbulence, and Risk Management*. Financial Analysts Journal.

---

## Disclaimer

This software is for **educational and research purposes only**. It does not constitute financial advice. Trading stocks involves risk. Past performance does not guarantee future results. Always do your own research before making investment decisions.
#   J o r d a n B e l f o R L  
 