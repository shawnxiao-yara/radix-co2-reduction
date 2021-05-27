"""Functions to extract features from a random list of samples."""
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.radix_co2_reduction.data import BANDS, load_pixel_data


def extract_vector(
    field_path: Path, band: str = "NDVI"
) -> Optional[np.ndarray]:  # TODO: Provide field-sample instead
    """Extract the combined vector of (down) sampled NDVI values of all field pixel values (averaged)."""
    assert band in BANDS

    # Load the data, if exists
    data, _ = load_pixel_data(
        field_path,
        downsample=10,
        window=30,
        remove_neg=True,  # Removes noisy samples
        remove_zero=False,  # Only complete vectors
    )
    if not data:
        return None
    data = np.asarray(data)
    data_comb = np.nan_to_num(np.nanmedian(np.where(data == 0, np.nan, data), axis=0))
    return data_comb[BANDS.index(band)]


def get_feature(vector: List[float]) -> List[float]:
    """
    Get the bucket feature corresponding the field-ID.

    This feature calculates for each bucket the min, mean, max, std, and slope-progress.
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
