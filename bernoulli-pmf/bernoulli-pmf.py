import numpy as np

def bernoulli_pmf_and_moments(x: list, p: float) -> dict:
    x_arr = np.array(x, dtype=float)
    pmf = np.where(x_arr == 1, p, np.where(x_arr == 0, 1.0 - p, 0.0))
    mean = float(p)
    variance = float(p * (1.0 - p))
    
    return {
        "pmf": pmf,
        "mean": mean,
        "variance": variance
    }