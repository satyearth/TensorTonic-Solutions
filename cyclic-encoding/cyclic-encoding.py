import math

def cyclic_encoding(values: list, period: float) -> list:
    encoded_values = []
    two_pi = 2 * math.pi

    for v in values:
        theta = two_pi * v / period
        encoded_values.append([math.sin(theta), math.cos(theta)])

    return encoded_values