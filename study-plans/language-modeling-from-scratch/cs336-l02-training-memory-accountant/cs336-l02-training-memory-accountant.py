def memory_accountant(
    param_shapes: list[list[int]], param_bytes_per_element: int,
    grad_bytes_per_element: int, activation_shapes: list[list[int]],
    activation_bytes_per_element: int, optimizer: str,
    optimizer_bytes_per_element: int,
) -> dict:

    def num_elements(shape):

        n = 1
        for dim in shape:
            n *= dim
        return n

    param_elements = sum(num_elements(shape) for shape in param_shapes)
    activation_elements = sum(
        num_elements(shape) for shape in activation_shapes
    )

    parameters = param_elements * param_bytes_per_element
    gradients = param_elements * grad_bytes_per_element
    activations = activation_elements * activation_bytes_per_element

    optimizer = optimizer.lower()

    if optimizer == "sgd":
        state_tensors = 0
    elif optimizer == "adagrad":
        state_tensors = 1
    elif optimizer == "adam":
        state_tensors = 2
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    optimizer_state = (
        param_elements
        * state_tensors
        * optimizer_bytes_per_element
    )

    total = parameters + gradients + activations + optimizer_state

    return {
        "parameters": parameters,
        "gradients": gradients,
        "activations": activations,
        "optimizer_state": optimizer_state,
        "total": total,
    }