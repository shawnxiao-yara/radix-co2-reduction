"""Shared scripts used in either pipeline."""
import json
from pathlib import Path
from typing import List, Tuple

from agoro_field_boundary_detector import FieldBoundaryDetectorInterface
from tqdm import tqdm

from src.co2_reduction.data import load_data
from src.co2_reduction.earth_engine import sample_fields


def sample(
    cache_folder: Path,
    coordinates: List[Tuple[float, float]],
    years: List[int],
    labels: List[bool],
    model_folder: Path = Path(__file__).parent / "../../models",
    mask_rcnn_tag: str = "mask_rcnn",
    min_samples: int = 5,
    n_pixels: int = 100,
) -> Tuple[List[Tuple[float, float]], List[int], List[bool]]:
    """
    Sample the fields under the given coordinates.

    :param cache_folder: Path towards a caching directory to store intermediate results (required due to large data size)
    :param coordinates: Coordinates (lat,lng) of the fields to sample
    :param years: Years in which to sample the corresponding fields
    :param labels: Tillage labels of the corresponding fields
    :param model_folder: Path towards the directory where the models are stored
    :param mask_rcnn_tag: Tag used to identify the Mask R-CNN model (found under the model_path folder)
    :param min_samples: Minimum number of samples/field required before considering the field for training
    :param n_pixels: Number of pixels to sample for each field
    :return: Tuple of successfully sampled coordinates with their corresponding year and label
    """
    # Load in all the field boundaries and filter out the samples that aren't recognised
    print("\nPredicting field boundaries...")
    model = FieldBoundaryDetectorInterface(
        model_path=model_folder / mask_rcnn_tag,
        new_session=False,  # Already done at this point
    )
    boundaries = []
    for lat, lng in tqdm(coordinates, "Predicting field boundaries..."):
        path = cache_folder / f"{lat}-{lng}"

        # Check if already exits
        if (path / "polygon.json").is_file():
            with open(path / "polygon.json", "r") as f:
                boundaries.append(json.load(f))
            continue

        # Compute boundary
        boundary = model(lat=lat, lng=lng)
        boundaries.append(boundary)

        # Write to cache
        if path is not None:
            path.mkdir(parents=True, exist_ok=True)
            with open(path / "polygon.json", "w") as f:
                json.dump(boundary, f)
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
        cache_path=cache_folder,
        n_pixels=n_pixels,
    )
    coordinates_f, years_f, labels_f = [], [], []
    for c, y, l in zip(coordinates, years, labels):  # noqa B007
        d = load_data(cache_folder / f"{c[0]}-{c[1]}")
        n_r = sum(v["R"] != [] for v in d.values())  # R,G,B,NIR,SWIR1,SWIR2
        n_sar_vv = sum(v["SAR_VV"] != [] for v in d.values())  # SAR_VV,SAR_VH
        if (n_r >= min_samples) and (n_sar_vv != 0):
            coordinates_f.append(c)
            years_f.append(y)
            labels_f.append(l)
    coordinates, years, labels = coordinates_f, years_f, labels_f
    print(f"Total of {len(labels)} fields successfully sampled")
    return coordinates, years, labels
