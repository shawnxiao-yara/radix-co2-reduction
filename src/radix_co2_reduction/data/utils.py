"""Utilisation functions."""
from datetime import datetime
from typing import Any, List

import numpy as np


def dma(
    date_idx: List[int],
    values: np.ndarray,
    target_dates: List[int],
    window: int = 30,
    min_val: int = 2,
) -> np.ndarray:
    """
    Distance based Moving Average.

    Weighted average based on distance to neighbouring values.

    :param date_idx: The offset date-indices corresponding the provided data
    :param values: Values corresponding each of the given dates
    :param target_dates: The offset date-indices to get calculate a date for
    :param window: Sliding window to indicate how many neighbouring days are taken into account (-window, window)
    :param min_val: Minimum number of values that should be present in the window before assigning zero
    :return: Array of indices
    """
    assert min_val >= 1
    zero_v = np.zeros((values.shape[0],))

    def get_avg(idx: int) -> Any:
        weights = np.asarray(
            [max(1 - abs(idx - date_idx[j]) / window, 0) for j in range(len(date_idx))]
        )
        if np.count_nonzero(weights) < min_val:
            return zero_v
        return (values * weights).sum(axis=1) / weights.sum()

    return np.transpose(np.vstack([get_avg(idx) for idx in target_dates]))


def datetime_to_int(date: str) -> int:
    """Transform the date (in YYYY-MM-DD format) to an integer on day-granularity."""
    return int(datetime.strptime(date, "%Y-%m-%d").timestamp() / 86400)  # 24 * 60 * 60
