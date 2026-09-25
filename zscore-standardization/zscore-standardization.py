import numpy as np

def zscore_standardize(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=float)

    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")

    mean = np.mean(X, axis=axis, keepdims=True)
    std = np.std(X, axis=axis, keepdims=True)

    # Standard deviation <= eps => entire slice becomes 0
    return np.divide(
        X - mean,
        std,
        out=np.zeros_like(X),
        where=std > eps
    )