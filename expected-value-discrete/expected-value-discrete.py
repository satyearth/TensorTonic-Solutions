import numpy as np

def expected_value_discrete(x: list, p: list) -> float:
    return float(np.dot(x, p))
    pass