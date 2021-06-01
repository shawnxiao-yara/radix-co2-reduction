"""Functions to extract features from a random list of samples."""
from typing import List

from scipy.stats import describe


def custom(values: List[float], perc: float = 0.1) -> List[float]:
    """Feature mapping best suited for tillage detection."""
    values = sorted(values)
    cut = round(perc * len(values))
    if cut:
        values = values[cut:-cut]
    if not values:
        return [0.0] * 4
    result = describe(values)
    return [
        result.minmax[0],
        values[len(values) // 2],
        result.mean,
        result.minmax[1],
    ]


# TODO: Deprecated
# def stat_describe(values: List[float], perc: float = 0.1) -> List[float]:
#     """Statistically describe the provided list of samples."""
#     values = sorted(values)
#     cut = round(perc * len(values))
#     if cut:
#         values = values[cut:-cut]
#     result = describe(values)
#     return [
#         result.minmax[0],
#         result.mean,
#         result.minmax[1],
#         result.variance if result.nobs > 1 else 0.0,
#         result.skewness,
#         result.kurtosis,
#     ]
#
#
# def indigo(values: List[float], min_perc=0.0, max_perc=0.9) -> List[float]:
#     """Extract the indigo-report's features."""
#     data = sorted(values)
#     min_perc_idx = round(min_perc * (len(data) - 1))
#     max_perc_idx = round(max_perc * (len(data) - 1))
#     return [data[min_perc_idx], data[max_perc_idx] - data[0]]
#
#
# def min_med_avg_max(values: List[float], perc: float = 0.1) -> List[float]:
#     """Create a vector of the minimum, median, average, and maximum value of the values-list."""
#     assert 0 <= perc < 0.5
#     values = sorted(values)
#     cut = round(perc * len(values))
#     if cut:
#         values = values[cut:-cut]
#     return [min(values), values[len(values) // 2], sum(values) / len(values), max(values)]
