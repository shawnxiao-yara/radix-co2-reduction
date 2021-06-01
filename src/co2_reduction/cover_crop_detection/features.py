"""Functions to extract features from a random list of samples."""
from typing import Dict, List, Optional

import numpy as np

from src.co2_reduction.data import BANDS, load_pixel_data


def ndvi_feature(
    sample: Dict[str, Dict[str, List[Optional[float]]]],
    year: int,
) -> List[float]:
    """Create a feature on the NDVI time-series data from the sample."""
    vector = extract_vector(sample, year)
    return get_feature(vector)  # type: ignore


def extract_vector(
    sample: Dict[str, Dict[str, List[Optional[float]]]],
    year: int,
    band: str = "NDVI",
) -> Optional[np.ndarray]:
    """Extract the combined vector of (down) sampled NDVI values of all field pixel values (averaged)."""
    assert band in BANDS

    # Load the data, if exists
    data = load_pixel_data(
        sample=sample,
        year=year,
        downsample=10,
        window=30,
        remove_neg=True,  # Removes noisy samples
        remove_zero=False,  # Only complete vectors
    )
    if not data:
        return None
    data = np.asarray(data)
    data_comb = np.nan_to_num(np.nanmedian(np.where(data == 0, np.nan, data), axis=0))
    return data_comb[BANDS.index(band)]  # type: ignore


def get_feature(vector: List[float]) -> List[float]:
    """
    Get the bucket feature corresponding the field-ID.

    This feature calculates for each bucket the min, mean, max, for the time series values and slope.
    """
    # Calculate slopes
    slopes = [
        vector[i + 1] - vector[i] if (vector[i + 1] != 0) and (vector[i] != 0) else 0
        for i in range(len(vector) - 1)
    ]

    # Remove zero values
    values = [v for v in vector if v != 0]
    slopes = [s for s in slopes if s != 0]

    # Create features
    return [
        min(values),
        sum(values) / len(values),
        max(values),
        min(slopes),
        sum(slopes) / len([s for s in slopes if s != 0]),
        max(slopes),
    ]
