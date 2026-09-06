def percent_change(series: list) -> list:

    result = []

    for i in range(1, len(series)):
        previous = series[i - 1]
        current = series[i]

        if previous == 0:
            result.append(0.0)
        else:
            result.append((current - previous) / previous)

    return result