def maxpool_forward(X, pool_size, stride):
    H = len(X)
    W = len(X[0])

    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1

    out = [[0] * W_out for _ in range(H_out)]

    for i in range(H_out):
        for j in range(W_out):
            max_val = float('-inf')

            for a in range(pool_size):
                for b in range(pool_size):
                    max_val = max(
                        max_val,
                        X[i * stride + a][j * stride + b]
                    )

            out[i][j] = max_val

    return out