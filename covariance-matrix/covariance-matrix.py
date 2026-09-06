import numpy as np

def covariance_matrix(X: list) -> np.ndarray:
    
    X = np.asarray(X, dtype=float)
    
    N = X.shape[0]
    
    mu = np.mean(X, axis=0)
    X_c = X - mu
    
    cov = (X_c.T @ X_c) / (N - 1)
    
    return cov