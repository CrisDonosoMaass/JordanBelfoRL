"""
models.py — DRL Model Management Module.

Handles initialization, configuration, training, saving, and loading
of the three actor-critic algorithms from Stable-Baselines3:
    - PPO  (Proximal Policy Optimization)     — Section 5.3
    - A2C  (Advantage Actor Critic)           — Section 5.1
    - DDPG (Deep Deterministic Policy Grad.)  — Section 5.2

Uses PyTorch backend with automatic CUDA/CPU detection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Type, Union

import torch
from stable_baselines3 import A2C, DDPG, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
import numpy as np

from src.environment import StockTradingEnv

logger = logging.getLogger("drl_ensemble.models")

# Map of supported algorithm names to SB3 classes
ALGORITHM_MAP: Dict[str, Type[BaseAlgorithm]] = {
    "ppo": PPO,
    "a2c": A2C,
    "ddpg": DDPG,
}


def create_model(
    algo_name: str,
    env: StockTradingEnv,
    config: dict,
    device: str = "auto",
) -> BaseAlgorithm:
    """
    Initialize a Stable-Baselines3 model with hyperparameters from config.

    Args:
        algo_name: Algorithm name ("ppo", "a2c", or "ddpg").
        env: The trading Gymnasium environment.
        config: Full YAML config dict.
        device: Compute device ("cuda", "cpu", or "auto").

    Returns:
        Initialized (untrained) SB3 model.
    """
    algo_name = algo_name.lower()
    if algo_name not in ALGORITHM_MAP:
        raise ValueError(
            f"Unknown algorithm '{algo_name}'. "
            f"Supported: {list(ALGORITHM_MAP.keys())}"
        )

    AlgoClass = ALGORITHM_MAP[algo_name]
    model_cfg = config["models"][algo_name].copy()

    # Extract total_timesteps (not a model kwarg)
    model_cfg.pop("total_timesteps", None)

    # Build policy kwargs (net_arch)
    policy_kwargs = model_cfg.pop("policy_kwargs", {})
    if "net_arch" in policy_kwargs:
        policy_kwargs["net_arch"] = policy_kwargs["net_arch"]

    # Common kwargs
    common_kwargs = {
        "policy": "MlpPolicy",
        "env": env,
        "device": device,
        "seed": config["training"]["seed"],
        "verbose": config["training"]["verbose"],
        "policy_kwargs": policy_kwargs,
    }

    # Algorithm-specific setup
    if algo_name == "ddpg":
        # DDPG requires action noise for exploration
        n_actions = env.action_space.shape[0]
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions),
        )
        common_kwargs["action_noise"] = action_noise

    # Merge hyperparameters
    common_kwargs.update(model_cfg)

    logger.info(
        "Creating %s model on device='%s' with policy_kwargs=%s",
        algo_name.upper(), device, policy_kwargs,
    )

    model = AlgoClass(**common_kwargs)
    return model


def train_model(
    model: BaseAlgorithm,
    total_timesteps: int,
    algo_name: str = "",
    save_path: Optional[str] = None,
    checkpoint_freq: int = 0,
) -> BaseAlgorithm:
    """
    Train a model for the specified number of timesteps.

    Args:
        model: Initialized SB3 model.
        total_timesteps: Number of timesteps to train.
        algo_name: Algorithm name for logging.
        save_path: Optional path to save the trained model.
        checkpoint_freq: Save checkpoints every N steps (0 = disabled).

    Returns:
        Trained model.
    """
    logger.info(
        "Training %s for %d timesteps...",
        algo_name.upper() or "model", total_timesteps,
    )

    # Train — use progress_bar only if tqdm/rich are available
    try:
        model.learn(total_timesteps=total_timesteps, progress_bar=True)
    except ImportError:
        logger.info("Progress bar unavailable (install tqdm+rich). Training without it...")
        model.learn(total_timesteps=total_timesteps, progress_bar=False)

    # Save if path provided
    if save_path:
        save_model(model, save_path)

    logger.info("Training %s complete.", algo_name.upper() or "model")
    return model


def save_model(model: BaseAlgorithm, path: str) -> None:
    """
    Save model weights to a .zip file.

    Args:
        model: Trained SB3 model.
        path: File path (with or without .zip extension).
    """
    path = str(path)
    if not path.endswith(".zip"):
        path += ".zip"

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save(path.replace(".zip", ""))  # SB3 adds .zip automatically
    logger.info("Model saved to %s", path)


def load_model(
    algo_name: str,
    path: str,
    env: Optional[StockTradingEnv] = None,
    device: str = "auto",
) -> BaseAlgorithm:
    """
    Load a trained model from disk.

    Args:
        algo_name: Algorithm name ("ppo", "a2c", "ddpg").
        path: Path to the saved model file.
        env: Optional environment to attach (for continued training).
        device: Compute device.

    Returns:
        Loaded model.
    """
    algo_name = algo_name.lower()
    if algo_name not in ALGORITHM_MAP:
        raise ValueError(f"Unknown algorithm '{algo_name}'.")

    AlgoClass = ALGORITHM_MAP[algo_name]

    # Remove .zip extension if present (SB3 expects path without it)
    path_clean = str(path).replace(".zip", "")

    logger.info("Loading %s model from %s", algo_name.upper(), path)
    model = AlgoClass.load(path_clean, env=env, device=device)
    return model


def predict_actions(
    model: BaseAlgorithm,
    state: "np.ndarray",
    deterministic: bool = True,
) -> "np.ndarray":
    """
    Get the model's action prediction for a given state.

    Args:
        model: Trained SB3 model.
        state: Current state observation.
        deterministic: If True, use deterministic policy (no exploration).

    Returns:
        Action array.
    """
    action, _ = model.predict(state, deterministic=deterministic)
    return action


def get_total_timesteps(config: dict, algo_name: str) -> int:
    """Extract total_timesteps for an algorithm from config."""
    return config["models"][algo_name.lower()]["total_timesteps"]
