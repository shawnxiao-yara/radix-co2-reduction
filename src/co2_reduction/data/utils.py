"""Utilisation functions."""
from datetime import datetime
from typing import Any, Dict, List, Optional

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


def disable_outliers(
    sample: Dict[str, Dict[str, List[Optional[float]]]],
    ratio: float = 0.05,
) -> Dict[str, Dict[str, List[Optional[float]]]]:
    """
    Disable the outliers by setting them to None.

    :param sample: Sample for which to disable the outliers (on day- and band-level)
    :param ratio: Symmetric ratio of not-None value outliers to disable
    """
    # Ignore if ratio is put to zero
    if ratio <= 0:
        return sample

    # Disable outliers in spectrum (0..ratio) and (1-ratio..1)
    for day, day_sample in sample.items():
        for band, values in day_sample.items():
            values_sort = sorted([v for v in values if v is not None])

            # Ignore if all-None already
            if not values_sort or round(ratio * len(values_sort)) == 0:
                continue

            # Check min and max threshold
            min_val = values_sort[round(ratio * len(values_sort))]
            max_val = values_sort[-round(ratio * len(values_sort))]
            values = [
                None if (v is None) or (v <= min_val) or (v >= max_val) else v for v in values
            ]
            sample[day][band] = values
    return sample
