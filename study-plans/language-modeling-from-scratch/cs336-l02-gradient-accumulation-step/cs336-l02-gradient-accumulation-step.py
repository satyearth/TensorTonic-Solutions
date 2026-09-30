import torch

def gradient_accumulation_step(
    param: torch.Tensor,
    microbatch_inputs: list[torch.Tensor],
    microbatch_targets: list[torch.Tensor],
    lr: float,
) -> dict:

    total_examples = sum(x.shape[0] for x in microbatch_inputs)

    full_grad = torch.zeros_like(param)

    for X, y in zip(microbatch_inputs, microbatch_targets):

        pred = X @ param

        error = pred - y

        grad_sum = 2 * (X.T @ error)

        full_grad += grad_sum

    full_grad /= total_examples

    new_param = param - lr * full_grad

    return {
        "new_param": new_param,
        "full_grad": full_grad,
    }