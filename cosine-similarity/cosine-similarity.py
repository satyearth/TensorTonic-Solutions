import numpy as np

def cosine_similarity(a: list, b: list) -> float:

    a_arr = np.array(a, dtype=np.float64)
    b_arr = np.array(b, dtype=np.float64)
    
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    
    dot_product = np.dot(a_arr, b_arr)
    
    return float(dot_product / (norm_a * norm_b))