"""Utilisation functions regarding the data."""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .utils import datetime_to_int, dma

BANDS = ("B", "G", "R", "NIR", "SWIR1", "SWIR2", "NDVI", "EVI", "NDTI")


def load_data(read_path: Path) -> Dict[str, Dict[str, List[Optional[float]]]]:
    """Load in the data, preference for sentinel2 > landsat8 > landsat7."""
    data = {}
    with open(read_path / "samples/landsat7.json", "r") as f:
        data.update(json.load(f))
    with open(read_path / "samples/landsat8.json", "r") as f:
        data.update(json.load(f))
    with open(read_path / "samples/sentinel2.json", "r") as f:
        data.update(json.load(f))
    return data


def get_label(field_path: Path) -> Any:
    """Get the field's labels."""
    with open(field_path / "meta.json", "r") as f:
        return json.load(f)["tillage"] != "No-Till"  # True if not tillage


def get_year(field_path: Path) -> Any:
    """Get the field's labels."""
    with open(field_path / "meta.json", "r") as f:
        return json.load(f)["year"]


# TODO: Change, or even remove?
def load_field_data(inp: Any) -> Tuple[Any, Optional[bool]]:  # noqa C901
    """Standalone function to load in a sample (multiprocessing purposes)."""
    read_path, cloud_filter = inp
    data: Dict[str, Any] = {}

    def add_data(new_data: Dict[str, Dict[str, List[float]]]) -> None:
        for date, sample in new_data.items():
            if cloud_filter is not None and cloud_filter(sample):
                continue
            if date in data:
                for band, values in sample.items():
                    data[date][band] += values
            else:
                data[date] = sample

    # Load in all the data
    with open(read_path / "samples/landsat7.json", "r") as f:
        add_data(json.load(f))
    with open(read_path / "samples/landsat8.json", "r") as f:
        add_data(json.load(f))
    with open(read_path / "samples/sentinel2.json", "r") as f:
        add_data(json.load(f))

    # Check if enough data samples remain, return empty result if not
    if len(data) < 5:
        return {}, None

    # Load in meta-data
    tillage = get_label(read_path)
    return data, tillage


def load_pixel_data(
    sample: Dict[str, Dict[str, List[Optional[float]]]],
    year: int,
    startdate_postfix: str = "-11-01",
    enddate_postfix: str = "-04-30",
    downsample: int = 10,
    window: int = 30,
    max_pixels: Optional[int] = None,
    remove_neg: bool = True,
    remove_zero: bool = False,
) -> List[np.ndarray]:
    """Standalone function to load in a sample (multiprocessing purposes)."""
    # Check if enough data samples remain, return empty result if not
    if len(sample) < 5:
        return []

    # Specify timeframe
    start_idx = datetime_to_int(f"{year - 1}{startdate_postfix}")
    end_idx = datetime_to_int(f"{year}{enddate_postfix}")

    # Add all pixel-level data
    data = process_sample_pixel(
        sample=sample,
        start_idx=start_idx,
        end_idx=end_idx,
        downsample=downsample,
        window=window,
        max_pixels=max_pixels,
        remove_neg=remove_neg,
        remove_zero=remove_zero,
    )

    # Load in meta-data
    return data


def process_sample_pixel(
    sample: Dict[str, Dict[str, List[Optional[float]]]],
    start_idx: int,
    end_idx: int,
    downsample: int = 10,
    window: int = 30,
    max_pixels: Optional[int] = None,
    outlier: float = 0.1,
    remove_neg: bool = True,
    remove_zero: bool = False,
) -> List[np.ndarray]:
    """
    Process the sample on pixel-level.

    :param sample: The sample to analyse
    :param start_idx: Startdate's index value
    :param end_idx: Enddate's index value
    :param downsample: Downsampling (expressed in number of days)
    :param window: DMA window
    :param max_pixels: Maximum number of pixels to consider
    :param outlier: Ratio of outliers to filter out
    :param remove_neg: Remove all negative (incl. zero) values, these are likely noisy/wrong inputs
    :param remove_zero: Remove sampled buckets with zero value (i.e. that had no input-data)
    """
    # Add all pixel-level data
    dates = sorted(sample.keys())
    n_pixels = len(list(sample.values())[0]["R"])
    temp_data = np.zeros((n_pixels, 9, len(dates)), dtype=np.float32)
    for i, date in enumerate(dates):
        for j, band in enumerate(BANDS):
            temp_data[:, j, i] = sample[date][band]

    # Remove pixel-values that are always None
    date_idx = np.asarray([datetime_to_int(date) for date in dates])
    target_dates = list(range(0, end_idx - start_idx + 1, downsample))
    temp_data = temp_data[~np.isnan(temp_data).all(axis=(1, 2)), :, :]

    # Remove dates where the mean is negative for at least one band
    if remove_neg:
        is_neg = np.nan_to_num(temp_data).mean(axis=0).min(axis=0) < 0
        if any(is_neg):
            temp_data = temp_data[:, :, ~is_neg]
            date_idx = date_idx[~is_neg]

    # Interpolate and downsample data
    data = []
    n_samples = min(temp_data.shape[0], max_pixels) if max_pixels else temp_data.shape[0]
    for i in range(n_samples):
        date_filter = ~np.isnan(temp_data[i]).any(axis=0)
        pixel_dates = [d - start_idx for d in date_idx[date_filter]]
        values = temp_data[i][:, date_filter]
        pixel_data = dma(
            date_idx=pixel_dates,
            values=values,
            target_dates=target_dates,
            window=window,
        )
        if remove_zero and (pixel_data == 0).any():
            continue
        data.append(pixel_data)

    # Clean out the outliers (those with the greatest 'outlier-score')
    if outlier > 0 and len(data) > 1 / outlier:
        data_avg = np.average(np.asarray(data), axis=0)
        scores = [np.abs(row - data_avg).sum() for row in data]
        max_score = sorted(scores)[-int(outlier * len(scores))]
        data = [d for d, s in zip(data, scores) if s < max_score]
    return data
