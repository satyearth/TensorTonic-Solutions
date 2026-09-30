import torch

def causal_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    B, Hq, S, D = q.shape
    Hkv = k.shape[1]
    G = Hq // Hkv

    kv_idx = torch.arange(Hq, device=q.device) // G
    k_grouped = k[:, kv_idx]
    v_grouped = v[:, kv_idx]

    scores = torch.matmul(q, k_grouped.transpose(-2, -1)) / (D ** 0.5)

    mask = torch.triu(
        torch.ones(S, S, device=q.device, dtype=torch.bool),
        diagonal=1,
    )
    scores = scores.masked_fill(mask, float("-inf"))

    attention = torch.softmax(scores, dim=-1)
    return torch.matmul(attention, v_grouped).to(dtype=q.dtype)