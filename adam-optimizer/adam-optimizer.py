import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    theta = np.asarray(param, dtype=float)
    g = np.asarray(grad, dtype=float)
    m_prev = np.asarray(m, dtype=float)
    v_prev = np.asarray(v, dtype=float)

    m_new = beta1 * m_prev + (1.0 - beta1) * g

    v_new = beta2 * v_prev + (1.0 - beta2) * (g**2)

    m_hat = m_new / (1.0 - beta1**t)
    v_hat = v_new / (1.0 - beta2**t)

    param_new = theta - lr * (m_hat / (np.sqrt(v_hat) + eps))

    return param_new, m_new, v_new