from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:

    arr = np.array(x, dtype=float)

    mean_val = float(np.mean(arr))
    
    median_val = float(np.median(arr))
    
    counts = Counter(x)
    max_freq = max(counts.values())
    
    mode_val = float(min(val for val, count in counts.items() if count == max_freq))
    
    return {
        "mean": mean_val,
        "median": median_val,
        "mode": mode_val
    }