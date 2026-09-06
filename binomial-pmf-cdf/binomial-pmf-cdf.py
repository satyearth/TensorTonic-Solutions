import math

def binomial_pmf_cdf(n: int, p: float, k: int) -> dict:
    
    if k < 0:
        return {"pmf": 0.0, "cdf": 0.0}
    if k > n:
        return {"pmf": 0.0, "cdf": 1.0}

    pmf = math.comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))

    cdf = sum(math.comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i)) for i in range(k + 1))
    
    return {
        "pmf": float(pmf),
        "cdf": float(cdf)
    }