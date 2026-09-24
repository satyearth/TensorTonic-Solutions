import numpy as np

def adamw_step(
    w: list,
    m: list,
    v: list,
    grad: list,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    weight_decay: float = 0.01,
    eps: float = 1e-8
) -> dict:

    w_arr = np.asarray(w, dtype=np.float64)
    m_arr = np.asarray(m, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    g_arr = np.asarray(grad, dtype=np.float64)

    new_m = beta1 * m_arr + (1.0 - beta1) * g_arr

    new_v = beta2 * v_arr + (1.0 - beta2) * (g_arr ** 2)

    new_w = w_arr - lr * (new_m / (np.sqrt(new_v) + eps)) - lr * weight_decay * w_arr

    return {
        "new_w": new_w,
        "new_m": new_m,
        "new_v": new_v,
    }