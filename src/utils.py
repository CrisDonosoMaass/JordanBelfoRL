"""
utils.py — Utilities for the DRL Ensemble Trading System.

Provides:
    - YAML configuration loading
    - Logging setup (console + file)
    - Device detection (CUDA / CPU)
    - Seed management for reproducibility
    - Path helpers
"""

import os
import logging
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml") -> dict:
    """Load the YAML configuration file and resolve special values."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Resolve "today" in trade_end
    if config["dates"].get("trade_end", "").lower() == "today":
        config["dates"]["trade_end"] = datetime.now().strftime("%Y-%m-%d")

    return config


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(config: dict, name: str = "drl_ensemble") -> logging.Logger:
    """
    Configure a logger with console and optional file handlers.

    Args:
        config: Loaded YAML config dict.
        name: Logger name.

    Returns:
        Configured Logger instance.
    """
    log_cfg = config.get("logging", {})
    level_str = log_cfg.get("level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    log_format = log_cfg.get(
        "log_format",
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    formatter = logging.Formatter(log_format)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_cfg.get("log_to_file", True):
        logs_dir = Path(config["paths"]["logs_dir"])
        logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            logs_dir / f"{name}_{timestamp}.log",
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def get_device(config: dict) -> str:
    """
    Determine the compute device based on config and hardware availability.

    Config value "auto" probes for CUDA, then falls back to CPU.

    Returns:
        "cuda" or "cpu".
    """
    requested = config.get("training", {}).get("device", "auto")

    if requested == "auto":
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            return "cuda"
        return "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"

    return requested


# ---------------------------------------------------------------------------
# Seed management
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set seeds across all libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Make CuDNN deterministic (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def ensure_dirs(config: dict) -> None:
    """Create all project directories specified in config['paths']."""
    for key, dir_path in config.get("paths", {}).items():
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    """Return the project root (parent of the src/ directory)."""
    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Timestamp helper
# ---------------------------------------------------------------------------

def timestamp_str() -> str:
    """Return a compact timestamp string for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
