import torch

def attention_scores(q: torch.Tensor, k: torch.Tensor, num_heads: int) -> torch.Tensor:

    B, Sq, D = q.shape
    Bk, Sk, Dk = k.shape

    if B != Bk:
        raise ValueError("q and k must have the same batch size")
    if D != Dk:
        raise ValueError("q and k must have the same model width")
    if D % num_heads != 0:
        raise ValueError("model width must be divisible by num_heads")

    dh = D // num_heads

    q_heads = q.reshape(B, Sq, num_heads, dh).transpose(1, 2)
    k_heads = k.reshape(B, Sk, num_heads, dh).transpose(1, 2)

    return torch.matmul(q_heads, k_heads.transpose(-2, -1)) / (dh ** 0.5)