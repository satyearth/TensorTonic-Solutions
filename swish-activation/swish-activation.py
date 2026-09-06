import numpy as np

def swish(x):

    x = np.asarray(x, dtype=float)

    sigmoid = np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x))
    )

    return x * sigmoid