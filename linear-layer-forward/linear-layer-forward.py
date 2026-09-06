def linear_layer_forward(X: list, W: list, b: list) -> list:
    n = len(X)
    d_in = len(X[0])
    d_out = len(W[0])

    Y = [[0.0 for _ in range(d_out)] for _ in range(n)]

    for i in range(n):
        for j in range(d_out):
            dot_sum = sum(X[i][k] * W[k][j] for k in range(d_in))
            Y[i][j] = dot_sum + b[j]

    return Y