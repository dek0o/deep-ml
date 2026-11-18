import numpy as np
from typing import List, Tuple


def k_fold_cross_validation(
    n_samples: int, k: int = 5, shuffle: bool = True
) -> List[Tuple[List[int], List[int]]]:
    if k < 2 or k > n_samples:
        raise ValueError(f"k={k} должно быть в [2, {n_samples}]")

    data = list(range(n_samples))
    if shuffle:
        np.random.shuffle(data)

    n_fold_samples, remainder = divmod(n_samples, k)

    bounds = [0] * (k + 1)
    for i in range(k):
        extra = 1 if remainder > 0 else 0 
        remainder -= 1
        # TODO: remainder равномерно размазывается на n первых массивов
        bounds[i + 1] = bounds[i] + n_fold_samples + extra
        

    splits: List[Tuple[List[int], List[int]]] = []
    for i in range(k):
        start, end = bounds[i], bounds[i + 1]
        test_idx = data[start:end]
        train_idx = data[:start] + data[end:]
        splits.append((train_idx, test_idx))
    return splits