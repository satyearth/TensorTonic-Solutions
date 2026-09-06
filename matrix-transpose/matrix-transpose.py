import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    
    if not A or not A[0]:
        return np.array([])

    rows = len(A)
    cols = len(A[0])

    transposed = [
        [A[row_idx][col_idx] for row_idx in range(rows)]
        for col_idx in range(cols)
    ]

    return np.array(transposed)