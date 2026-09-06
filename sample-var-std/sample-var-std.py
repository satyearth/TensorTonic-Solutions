import numpy as np

def sample_var_std(x: list) -> dict:

    arr = np.array(x, dtype=float)

    variance = float(np.var(arr, ddof=1))
    std_dev = float(np.std(arr, ddof=1))

    return {"variance": variance, "standard_deviation": std_dev}