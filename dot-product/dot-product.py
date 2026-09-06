import numpy as np

def dot_product(x: list, y: list) -> float:
    return float(sum(a * b for a, b in zip(x, y)))
    pass