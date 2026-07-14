"""
Model loading with Streamlit caching.

Two completely separate model registries:
  AGGREGATE_MODELS → saved_models/aggregate/*.keras   (22 features, seq_len=60)
  STORE_MODELS     → saved_models/store/*.keras       (30 features, seq_len=60)

Filenames match exactly what is on disk (from the user's trained models).
"""

import os
import streamlit as st

# Import custom layers BEFORE loading models
# This registers them with Keras so they can be deserialized
from .nhits_layers import NHiTSBlock, NHiTS

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")

# ─────────────────────────────────────────────────────────────
# AGGREGATE REGISTRY
# Display Name  →  filename inside saved_models/aggregate/
# ─────────────────────────────────────────────────────────────
AGGREGATE_MODEL_FILES = {
    "FNN":              "fnn_model.keras",
    "GRU":              "gru_model.keras",
    "N-HiTS":           "nhits_model.keras",
    "Quantile FNN":     "fnn_quantile_model.keras",
    "Quantile RNN":     "rnn_quantile_model.keras",
    "Quantile GRU":     "gru_quantile_model.keras",
    "Quantile N-HiTS":  "nhits_quantile_model.keras",
}

# ─────────────────────────────────────────────────────────────
# STORE REGISTRY
# These are SEPARATE models trained on store-level features (30 cols).
# Place them in saved_models/store/ after training.
# ─────────────────────────────────────────────────────────────
STORE_MODEL_FILES = {
    "FNN":              "fnn_model.keras",
    "GRU":              "gru_model.keras",
    "N-HiTS":           "nhits_model.keras",
    "Quantile FNN":     "fnn_quantile_model.keras",
    "Quantile RNN":     "rnn_quantile_model.keras",
    "Quantile GRU":     "gru_quantile_model.keras",
    "Quantile N-HiTS":  "nhits_quantile_model.keras",
}

QUANTILE_MODEL_NAMES = {"Quantile FNN", "Quantile RNN", "Quantile GRU", "Quantile N-HiTS"}


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _model_path(pipeline: str, filename: str) -> str:
    return os.path.join(BASE_DIR, pipeline, filename)


def available_models(pipeline: str = "aggregate") -> list:
    """Return models whose .keras file actually exists on disk."""
    registry = AGGREGATE_MODEL_FILES if pipeline == "aggregate" else STORE_MODEL_FILES
    present = [
        name for name, fname in registry.items()
        if os.path.exists(_model_path(pipeline, fname))
    ]
    # Fall back to full list so UI renders even before models are placed
    return present if present else list(registry.keys())


def is_nhits_model(model_name: str) -> bool:
    """Check if model is N-HiTS (which has loading issues)."""
    return "N-HiTS" in model_name or "nhits" in model_name.lower()


def get_nhits_warning() -> str:
    """Get warning message for N-HiTS models."""
    return (
        "⚠️  N-HiTS models require special handling due to custom layer architecture. "
        "They may not load correctly in the current environment. "
        "Please use GRU, FNN, or RNN models instead. "
        "To enable N-HiTS: contact support or re-train the models with proper serialization."
    )


def store_models_available() -> bool:
    """True if at least one store model .keras file exists."""
    return any(
        os.path.exists(_model_path("store", fname))
        for fname in STORE_MODEL_FILES.values()
    )


def is_quantile_model(model_name: str) -> bool:
    return model_name in QUANTILE_MODEL_NAMES


@st.cache_resource(show_spinner=False)
def load_model(pipeline: str, model_name: str):
    """Load and cache a Keras model. pipeline='aggregate' or 'store'."""
    try:
        from tensorflow import keras
    except ImportError:
        raise ImportError("TensorFlow is required: pip install tensorflow")

    registry = AGGREGATE_MODEL_FILES if pipeline == "aggregate" else STORE_MODEL_FILES
    filename = registry.get(model_name)
    if filename is None:
        raise ValueError(f"Unknown model '{model_name}' for pipeline '{pipeline}'")

    path = _model_path(pipeline, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            f"Train your {pipeline} models and place them in saved_models/{pipeline}/"
        )

    try:
        # Custom objects dict for N-HiTS layers
        custom_objects = {
            'NHiTSBlock': NHiTSBlock,
            'NHiTS': NHiTS,
        }
        return keras.models.load_model(path, custom_objects=custom_objects, compile=False)
    except Exception as e:
        error_msg = str(e)
        if "NHiTSBlock" in error_msg or "NHiTS" in error_msg:
            raise TypeError(
                f"Failed to load N-HiTS model.\n"
                f"Error: {error_msg}\n"
                f"Note: N-HiTS requires the exact custom layer implementation from training."
            ) from e
        raise
