def flop_estimator(matmuls: list[list[int]], attention_flops: int = 0) -> dict:

    forward_flops = int(attention_flops)

    for B, D, K in matmuls:
        forward_flops += 2 * B * D * K

    backward_flops = 2 * forward_flops
    total_flops = forward_flops + backward_flops

    return {
        "forward_flops": forward_flops,
        "backward_flops": backward_flops,
        "total_flops": total_flops,
    }