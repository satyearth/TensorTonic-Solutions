import numpy as np

def matrix_normalization(matrix: list, axis=None, norm_type: str = "l2") -> np.ndarray:
    
    mat = np.asarray(matrix, dtype=float)
    norm = norm_type.lower()
    
    if norm == "l1":
        denom = np.sum(np.abs(mat), axis=axis, keepdims=True)
    elif norm == "l2":
        denom = np.sqrt(np.sum(mat ** 2, axis=axis, keepdims=True))
    elif norm in ("max", "linf", "infinity"):
        denom = np.max(np.abs(mat), axis=axis, keepdims=True)
    else:
        raise ValueError(f"Unsupported norm_type: {norm_type}")

    result = np.zeros_like(mat)
    nonzero_mask = denom > 0
    np.divide(mat, denom, out=result, where=nonzero_mask)
    
    return result