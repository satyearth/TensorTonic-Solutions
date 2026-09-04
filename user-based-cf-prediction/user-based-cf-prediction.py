def user_based_cf_prediction(similarities: list, ratings: list) -> float:
    
    weighted_sum = 0.0
    sim_sum = 0.0

    for s, r in zip(similarities, ratings):
        if s > 0:
            weighted_sum += s * r
            sim_sum += s

    if sim_sum == 0.0:
        return 0.0

    return round(weighted_sum / sim_sum, 6)