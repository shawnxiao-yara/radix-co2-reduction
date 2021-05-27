"""Predict field boundaries for the given coordinates."""
import json
import os
from pathlib import Path
from random import getrandbits
from typing import Any, List, Optional, Tuple

from tqdm import tqdm

from src.radix_co2_reduction.earth_engine.datasets import NaipCollection
from src.radix_co2_reduction.earth_engine.utils import (
    create_bounding_box,
    download_as_png,
    to_polygon,
)
from src.radix_co2_reduction.field_detection import adjust_polygon, load_model, predict_im_polygon

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def get_field_polygon(
    coordinate: Tuple[float, float],
    model: Optional[Any] = None,
) -> List[Tuple[float, float]]:
    """
    Extract the field-boundaries covering the provided coordinate.

    :param coordinate: The coordinate (lat,lng) for which to predict field boundaries
    :param model: The Mask-RCNN model used to predict
    :return: Field boundaries as a list of coordinates (lat,lng) or None if no field is detected
    """
    # Assure model is loaded
    if model is None:
        model = load_model()

    # Specify temporary file to export field-image to
    im_f = Path.cwd() / f"{getrandbits(128)}.png"

    # Create bounding-box around given coordinate
    bounding_box = to_polygon(
        create_bounding_box(
            lng=coordinate[1],
            lat=coordinate[0],
            offset=1000,
        )
    )

    # Load in the dataset used to detect the field's boundaries
    coll = NaipCollection()
    coll.load_collection(
        region=bounding_box,
        startdate="2017-01-01",  # Assumption: Field parcels remain the same over time
        enddate="2020-12-31",
        return_masked=False,  # Only interested in raw images
    )

    # Create an image of the coordinate
    download_as_png(
        im=coll.collection.mosaic(),
        vis_param=coll.vis_param,
        region=bounding_box,
        write_path=im_f,
        dimensions=(1000, 1000),
    )

    # Predict the surrounding polygon
    im_polygon = predict_im_polygon(
        model=model,
        im_path=im_f,
    )
    if not im_polygon:
        im_f.unlink(missing_ok=True)
        return []

    # Adjust to (lat,lng)
    polygon = adjust_polygon(
        coordinate=coordinate,
        im_path=im_f,
        im_polygon=im_polygon,
    )

    # Remove the temporary file and return the result
    im_f.unlink(missing_ok=True)
    return polygon  # type: ignore


def extract_field_boundaries(
    coordinates: List[Tuple[float, float]],
    cache_path: Path,
    model_path: Path = Path(__file__).parent / "../../models",
) -> List[List[Tuple[float, float]]]:
    """
    Detect the field-boundaries for all the fields specified by the coordinate, or empty list if no field is found.

    :param coordinates: The coordinates (lat,lng) for which to predict a field-boundary
    :param cache_path: Path to data cache
    :param model_path: Path where the Mask-RCNN model is stored
    :return: List of field boundaries as a list of coordinates (lat,lng)
    """
    # Load in the Mask-RCNN model
    model = load_model(model_path=model_path)

    # Create field-polygons for all the coordinates
    boundaries = []
    for coor in tqdm(coordinates, desc="Predicting field boundaries.."):
        path = cache_path / f"{coor[0]}-{coor[1]}" if cache_path is not None else None

        # Check if already exits
        if path is not None and (path / "polygon.json").is_file():
            with open(path / "polygon.json", "r") as f:
                boundaries.append(json.load(f))
            continue

        # Compute boundary
        boundary = get_field_polygon(model=model, coordinate=coor)
        boundaries.append(boundary)

        # Write to cache
        if path is not None:
            path.mkdir(parents=True, exist_ok=True)
            with open(path / "polygon.json", "w") as f:
                json.dump(boundary, f)
    return boundaries
