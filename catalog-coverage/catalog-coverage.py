def catalog_coverage(recommendations: list, n_items: int) -> float:
    
    if n_items == 0:
        return 0.0

    unique_items = set(item for rec_list in recommendations for item in rec_list)
    return len(unique_items) / n_items