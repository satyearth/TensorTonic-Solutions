import torch

def rotary_embed(
    q: torch.Tensor, k: torch.Tensor,
    positions: torch.Tensor, inv_freq: torch.Tensor,
) -> dict:

    if positions.ndim == 1:

        theta = positions[:, None] * inv_freq[None, :]
        theta = theta[None, None, :, :]
    else:
        
        theta = positions[:, :, None] * inv_freq[None, None, :]
        theta = theta[:, None, :, :]
    cos_theta = torch.cos(theta).to(dtype=q.dtype, device=q.device)
    sin_theta = torch.sin(theta).to(dtype=q.dtype, device=q.device)

    def rotate(x: torch.Tensor) -> torch.Tensor:

        x_pairs = x.reshape(*x.shape[:-1], -1, 2)

        even = x_pairs[..., 0]
        odd = x_pairs[..., 1]

        rotated_even = even * cos_theta - odd * sin_theta
        rotated_odd = even * sin_theta + odd * cos_theta

        return torch.stack((rotated_even, rotated_odd), dim=-1).reshape_as(x)

    return {
        "q_rotated": rotate(q),
        "k_rotated": rotate(k),
    }