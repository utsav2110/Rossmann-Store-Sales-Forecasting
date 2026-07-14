"""
Inference logic for both point and quantile models.
Handles flat (FNN) and sequential (GRU/RNN/N-HiTS) input shapes.
"""

import numpy as np
from typing import Dict

from .load_models import load_model, is_quantile_model
from utils.scaler import inverse_scale_target


def predict(
    pipeline: str,          # "aggregate" | "store"
    model_name: str,
    sequence: np.ndarray,   # already shaped correctly by build_sequence()
    target_scaler,
    future_features: np.ndarray = None,  # optional: shape (1, n_future_feats) for dual-input models
) -> Dict[str, float]:
    """
    Run inference and return a results dict.

    Parameters
    ----------
    pipeline : str
        "aggregate" or "store"
    model_name : str
        Model name (e.g., "GRU", "N-HiTS", "FNN")
    sequence : np.ndarray
        Sequence input, already shaped by build_sequence()
    target_scaler
        Target scaler for inverse transformation
    future_features : np.ndarray, optional
        Future features for dual-input models, shape (1, n_future_feats).
        For FNN (Sequential): concatenates into a single flat input
        For GRU/RNN/N-HiTS (Functional): passes [sequence, future_features] separately

    Returns
    -------
    Dict[str, float]
        Normal model  → {"prediction": float}
        Quantile model → {"lower": float, "median": float, "upper": float}
    """
    if sequence is None or sequence.size == 0:
        raise ValueError("Input sequence is empty")
    
    model = load_model(pipeline, model_name)
    
    # Handle FNN vs dual-input models differently
    if future_features is not None:
        # FNN models are Sequential and need concatenated flat input
        if "FNN" in model_name or "fnn" in model_name:
            # Concatenate: flatten future_features and append to sequence
            model_input = np.concatenate([sequence, future_features], axis=1)
            raw = model.predict(model_input, verbose=0)
        else:
            # GRU/RNN/N-HiTS are Functional models with dual inputs
            raw = model.predict([sequence, future_features], verbose=0)
    else:
        raw = model.predict(sequence, verbose=0)
    
    if raw is None or raw.size == 0:
        raise ValueError("Model returned empty output")

    if is_quantile_model(model_name):
        return _quantile_result(raw, target_scaler)
    else:
        return _point_result(raw, target_scaler)


def _point_result(raw: np.ndarray, target_scaler) -> Dict[str, float]:
    val = float(max(inverse_scale_target(target_scaler, raw)[0], 0))
    return {"prediction": val}


def _quantile_result(raw: np.ndarray, target_scaler) -> Dict[str, float]:
    if raw.shape[-1] >= 3:
        low    = float(max(inverse_scale_target(target_scaler, raw[:, 0:1])[0], 0))
        median = float(max(inverse_scale_target(target_scaler, raw[:, 1:2])[0], 0))
        high   = float(max(inverse_scale_target(target_scaler, raw[:, 2:3])[0], 0))
    else:
        point  = float(max(inverse_scale_target(target_scaler, raw)[0], 0))
        low, median, high = point * 0.90, point, point * 1.10

    return {
        "lower":  min(low, median),
        "median": median,
        "upper":  max(high, median),
    }


def format_inr(value: float) -> str:
    """Format as Euros with Lakh/Crore suffix."""
    if value >= 1_00_00_000:
        return f"€ {value / 1_00_00_000:.2f} Cr"
    elif value >= 1_00_000:
        return f"€ {value / 1_00_000:.2f} L"
    else:
        return f"€ {value:,.0f}"
