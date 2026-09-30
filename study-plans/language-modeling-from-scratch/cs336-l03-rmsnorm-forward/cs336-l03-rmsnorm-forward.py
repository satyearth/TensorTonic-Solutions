import torch

def rmsnorm(x: torch.Tensor, g: torch.Tensor, epsilon: float) -> torch.Tensor:

    mean_square = x.pow(2).mean(dim=-1, keepdim=True)

    rms_inv = torch.rsqrt(mean_square + epsilon)

    return x * rms_inv * g