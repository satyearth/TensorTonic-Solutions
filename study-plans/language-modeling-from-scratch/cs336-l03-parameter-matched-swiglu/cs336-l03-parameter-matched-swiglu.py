import torch

def parameter_matched_swiglu(
    x: torch.Tensor, w_g: torch.Tensor, w_v: torch.Tensor,
    w_o: torch.Tensor, base_params: int,
) -> dict:

    d = x.shape[-1]
    H = w_g.shape[1]

    h = min(H, max(1, int(base_params / (3 * d) + 0.5)))

    Wg = w_g[:, :h]
    Wv = w_v[:, :h]
    Wo = w_o[:h, :]

    gate = torch.nn.functional.silu(x @ Wg)
    value = x @ Wv
    hidden = gate * value

    output = hidden @ Wo

    return {
        "output": output,
        "hidden_width": h,
        "parameter_count": 3 * d * h,
    }