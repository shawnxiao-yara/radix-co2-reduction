"""Process the samples."""
from datetime import datetime, timedelta
from random import gauss, randint
from typing import Any, Dict, List

import numpy as np


def med(values: List[float]) -> float:
    """Get the median of the list of values."""
    values = sorted(values)
    return values[len(values) // 2]


def avg(values: List[float], perc: float = 0.1) -> float:
    """Get the average value, after cutting of the bottom and top percentiles."""
    values = sorted(values)
    cut = round(perc * len(values))
    values = values[cut:-cut]
    return sum(values) / max(len(values), 1)


def manipulate_values(values: List[float], normalise: bool = True) -> List[float]:
    """Manipulate the given list of samples."""
    std = float(np.std(values))
    if normalise:
        return [max(min(gauss(v, std / 5), 1), 0) for v in values]  # noqa S311
    else:
        return [gauss(v, std / 5) for v in values]  # noqa S311


def manipulate_time(samples: Dict[str, Any], day_offset: int = 5) -> Dict[str, Any]:
    """Manipulate the time-stamps of the given samples."""
    result = {}
    for k, v in samples.items():
        k_d = datetime.strptime(k, "%Y-%m-%d")
        delta = timedelta(days=randint(-day_offset, day_offset))  # noqa S311
        result[(k_d + delta).strftime("%Y-%m-%d")] = v
    return result
