"""
Scaler utilities — load and apply sklearn scalers from .pkl files.
Two separate scaler pairs exist:
  saved_models/aggregate/ → 22-feature aggregate pipeline
  saved_models/store/     → 30-feature store pipeline
"""

import os
import pickle
import numpy as np

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")


def load_scalers(pipeline: str = "aggregate"):
    """
    Load (feature_scaler, target_scaler) for the given pipeline.

    Parameters
    ----------
    pipeline : "aggregate" | "store"
    """
    folder = os.path.join(BASE_DIR, pipeline)
    fs_path = os.path.join(folder, "feature_scaler.pkl")
    ts_path = os.path.join(folder, "target_scaler.pkl")

    if not os.path.exists(fs_path):
        raise FileNotFoundError(
            f"feature_scaler.pkl not found in saved_models/{pipeline}/\n"
            f"Expected path: {fs_path}"
        )
    if not os.path.exists(ts_path):
        raise FileNotFoundError(
            f"target_scaler.pkl not found in saved_models/{pipeline}/\n"
            f"Expected path: {ts_path}"
        )

    with open(fs_path, "rb") as f:
        feature_scaler = pickle.load(f)
    with open(ts_path, "rb") as f:
        target_scaler = pickle.load(f)

    return feature_scaler, target_scaler


def scale_features(feature_scaler, features_2d: np.ndarray) -> np.ndarray:
    return feature_scaler.transform(features_2d)


def inverse_scale_target(target_scaler, pred_scaled: np.ndarray) -> np.ndarray:
    flat = pred_scaled.reshape(-1, 1)
    return target_scaler.inverse_transform(flat).flatten()
