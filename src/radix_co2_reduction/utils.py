"""Shared scripts used in either pipeline."""
from pathlib import Path
from typing import List, Tuple

from src.radix_co2_reduction.data import load_data
from src.radix_co2_reduction.earth_engine import sample_fields
from src.radix_co2_reduction.field_detection import extract_field_boundaries


def sample(
    cache_path: Path,
    coordinates: List[Tuple[float, float]],
    years: List[int],
    labels: List[bool],
    model_path: Path = Path(__file__).parent / "../../models",
    min_samples: int = 5,
    n_pixels: int = 100,
) -> Tuple[List[Tuple[float, float]], List[int], List[bool]]:
    """
    Sample the fields under the given coordinates.

    :param cache_path: Path towards a caching directory to store intermediate results (required due to large data size)
    :param coordinates: Coordinates (lat,lng) of the fields to sample
    :param years: Years in which to sample the corresponding fields
    :param labels: Tillage labels of the corresponding fields
    :param model_path: Path towards the directory where the models are stored
    :param min_samples: Minimum number of samples/field required before considering the field for training
    :param n_pixels: Number of pixels to sample for each field
    :return: Tuple of successfully sampled coordinates with their corresponding year and label
    """
    # Load in all the field boundaries and filter out the samples that aren't recognised
    print("\nPredicting field boundaries...")
    boundaries = extract_field_boundaries(
        coordinates=coordinates,
        model_path=model_path,
        cache_path=cache_path,
    )
    assert len(boundaries) == len(coordinates)
    coordinates_f, years_f, labels_f, boundaries_f = [], [], [], []
    for c, y, l, b in zip(coordinates, years, labels, boundaries):
        if b:
            coordinates_f.append(c)
            years_f.append(y)
            labels_f.append(l)
            boundaries_f.append(b)
    coordinates, years, labels, boundaries = coordinates_f, years_f, labels_f, boundaries_f
    print(f"Total of {len(labels)} field boundaries successfully extracted")

    # Sample fields using Google Earth Engine
    print("\nSampling fields...")
    sample_fields(
        coordinates=coordinates,
        boundaries=boundaries,
        years=years,
        cache_path=cache_path,
    n_pixels= n_pixels,
    )
    coordinates_f, years_f, labels_f = [], [], []
    for c, y, l in zip(coordinates, years, labels):  # noqa B007
        d = load_data(cache_path / f"{c[0]}-{c[1]}")
        n_r = sum(v["R"] != [] for v in d.values())  # R,G,B,NIR,SWIR1,SWIR2
        n_sar_vv = sum(v["SAR_VV"] != [] for v in d.values())  # SAR_VV,SAR_VH
        if (n_r >= min_samples) and (n_sar_vv != 0):
            coordinates_f.append(c)
            years_f.append(y)
            labels_f.append(l)
    coordinates, years, labels = coordinates_f, years_f, labels_f
    print(f"Total of {len(labels)} fields successfully sampled")
    return coordinates, years, labels
