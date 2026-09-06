def classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    average: str = "micro",
    pos_label: int = 1,
) -> dict:

    n = len(y_true)

    correct_total = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = correct_total / n if n > 0 else 0.0

    classes = sorted(list(set(y_true) | set(y_pred)))

    tp, fp, fn, support = {}, {}, {}, {}
    for c in classes:
        tp[c] = 0
        fp[c] = 0
        fn[c] = 0
        support[c] = 0

    for yt, yp in zip(y_true, y_pred):
        support[yt] += 1
        if yt == yp:
            tp[yt] += 1
        else:
            fp[yp] += 1
            fn[yt] += 1

    def calc_p_r_f1(tp_val: int, fp_val: int, fn_val: int) -> tuple[float, float, float]:
        p = tp_val / (tp_val + fp_val) if (tp_val + fp_val) > 0 else 0.0
        r = tp_val / (tp_val + fn_val) if (tp_val + fn_val) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1

    if average == "micro":
        total_tp = sum(tp.values())
        total_fp = sum(fp.values())
        total_fn = sum(fn.values())
        precision, recall, f1 = calc_p_r_f1(total_tp, total_fp, total_fn)

    elif average == "binary":
        c_tp = tp.get(pos_label, 0)
        c_fp = fp.get(pos_label, 0)
        c_fn = fn.get(pos_label, 0)
        precision, recall, f1 = calc_p_r_f1(c_tp, c_fp, c_fn)

    elif average == "macro":
        p_list, r_list, f1_list = [], [], []
        for c in classes:
            p_c, r_c, f1_c = calc_p_r_f1(tp[c], fp[c], fn[c])
            p_list.append(p_c)
            r_list.append(r_c)
            f1_list.append(f1_c)

        num_classes = len(classes)
        precision = sum(p_list) / num_classes if num_classes > 0 else 0.0
        recall = sum(r_list) / num_classes if num_classes > 0 else 0.0
        f1 = sum(f1_list) / num_classes if num_classes > 0 else 0.0

    elif average == "weighted":
        precision, recall, f1 = 0.0, 0.0, 0.0
        for c in classes:
            p_c, r_c, f1_c = calc_p_r_f1(tp[c], fp[c], fn[c])
            weight = support[c] / n if n > 0 else 0.0
            precision += p_c * weight
            recall += r_c * weight
            f1 += f1_c * weight

    return {
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }