"""
Sequence creation.
FNN models expect flat (1, seq_len * n_features).
GRU / RNN / N-HiTS expect 3-D (1, seq_len, n_features).
"""

import numpy as np

SEQUENCE_LENGTH = 60


def build_sequence(features_scaled: np.ndarray, model_name_or_type: str, seq_len: int = SEQUENCE_LENGTH) -> np.ndarray:
    """
    Return the correct input tensor shape for the given model type.

    Parameters
    ----------
    features_scaled : np.ndarray  shape (N, F)
    model_name_or_type : str like "FNN" or "Quantile FNN" or "fnn"
    seq_len         : window length (default 60)

    Returns
    -------
    np.ndarray
        shape (1, seq_len * F)  for FNN
        shape (1, seq_len, F)   for GRU / RNN / N-HiTS
    """
    # Convert display name to architecture type if needed
    model_type = model_type_from_name(model_name_or_type)
    
    n, f = features_scaled.shape
    if n >= seq_len:
        window = features_scaled[-seq_len:]
    else:
        pad    = np.zeros((seq_len - n, f), dtype=np.float32)
        window = np.vstack([pad, features_scaled])

    if model_type == "fnn":
        return window.flatten()[np.newaxis, :]      # (1, seq_len * F)
    else:
        return window[np.newaxis, :, :]             # (1, seq_len, F)


def model_type_from_name(model_name: str) -> str:
    """Derive the architecture type from the display name."""
    name = model_name.lower()
    if "fnn" in name:
        return "fnn"
    if "gru" in name:
        return "gru"
    if "rnn" in name:
        return "rnn"
    if "nhits" in name or "n-hits" in name or "n_hits" in name:
        return "nhits"
    return "gru"   # safe default for 3-D models
