"""Google Earth Engine methods to extract data."""
import json
from pathlib import Path
from typing import List, Tuple

import ee
from tqdm import tqdm
from time import sleep
from src.radix_co2_reduction.earth_engine.datasets import (
    CroplandCollection,
    Landsat7Collection,
    Landsat8Collection,
    Sentinel1Collection,
    Sentinel2Collection,
)
from src.radix_co2_reduction.earth_engine.utils import create_polygon


def sample_field(
    coordinate: Tuple[float, float],
    boundary: List[Tuple[float, float]],
    cache_path: Path,
    year: int,
    startdate_postfix: str = "-11-01",
    enddate_postfix: str = "-04-30",
    n_pixels: int = 100,
    overwrite: bool = False,
) -> None:
    """
    Sample the field, as specified by its boundary polygon.

    :param coordinate: Field coordinate (lat,lng)
    :param boundary: Field's boundary polygon as a list of (lat,lng) coordinates
    :param cache_path: Directory of data cache
    :param startdate_postfix: Starting date postfix of the previous year (month and day)
    :param enddate_postfix: Ending date postfix of the previous year (month and day)
    :param year: The year of interest
    :param n_pixels: Number of pixels to sample per time-period (day) over the given field
    :param overwrite: Overwrite previously stored data
    """
    # Write down in folder (cache)
    write_f = cache_path / f"{coordinate[0]}-{coordinate[1]}/samples"
    write_f.mkdir(parents=True, exist_ok=True)
    if not overwrite and (write_f / "sentinel2.json").is_file():
        return

    # Create an Earth Engine polygon for the field-coordinates
    boundary = [(b, a) for a, b in boundary]  # Input should be in (lng,lat)
    field_polygon = create_polygon([boundary])

    # Load in all relevant datasets over the specified timeframe
    startdate = f"{year - 1}{startdate_postfix}"
    enddate = f"{year}{enddate_postfix}"

    # Landsat 7 data: https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LE07_C01_T1_SR
    landsat7 = Landsat7Collection()
    landsat7.load_collection(
        region=field_polygon,
        startdate=startdate,
        enddate=enddate,
        filter_clouds=True,
        filter_perc=0.75,
    )
    # Landsat 8 data: https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C01_T1_SR
    landsat8 = Landsat8Collection()
    landsat8.load_collection(
        region=field_polygon,
        startdate=startdate,
        enddate=enddate,
        filter_clouds=True,
        filter_perc=0.75,
    )
    # Sentinel-1 SAR GRD data: https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD
    sentinel1 = Sentinel1Collection()
    sentinel1.load_collection(
        region=field_polygon,
        startdate=startdate,
        enddate=enddate,
    )
    # Sentinel-2 L2A data: https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR
    sentinel2 = Sentinel2Collection()
    sentinel2.load_collection(
        region=field_polygon,
        startdate=startdate,
        enddate=enddate,
        filter_clouds=True,
        filter_perc=0.75,
    )

    # Add Cropland data layer to all image collections and use it as a mask
    cropland = CroplandCollection()
    cropland.load_collection(
        region=field_polygon,
        startdate=f"{year}-01-01",
        enddate=f"{year}-12-31",
    )
    cropland_im = cropland.collection.first().select("cropland")
    landsat7.mask_cropland(cropland_im=cropland_im, region=field_polygon)
    landsat8.mask_cropland(cropland_im=cropland_im, region=field_polygon)
    sentinel1.mask_cropland(cropland_im=cropland_im, region=field_polygon)
    sentinel2.mask_cropland(cropland_im=cropland_im, region=field_polygon)

    # Collect pixels to sample over
    pixels = ee.FeatureCollection.randomPoints(
        region=field_polygon,
        points=n_pixels,
        seed=42,
        maxError=50,
    )

    # Sample the collections
    s_landsat7 = landsat7.sample(pixels=pixels)
    with open(write_f / "landsat7.json", "w") as f:
        json.dump(s_landsat7, f)
    s_landsat8 = landsat8.sample(pixels=pixels)
    with open(write_f / "landsat8.json", "w") as f:
        json.dump(s_landsat8, f)
    s_sentinel1 = sentinel1.sample(pixels=pixels)
    with open(write_f / "sentinel1.json", "w") as f:
        json.dump(s_sentinel1, f)
    s_sentinel2 = sentinel2.sample(pixels=pixels)
    with open(write_f / "sentinel2.json", "w") as f:
        json.dump(s_sentinel2, f)
    sleep(5)  # Prevent getting blocked by GEE
    
    # TODO: Intermediate check, might be removed later on
    shapes = set()
    for coll in (s_landsat7, s_landsat8, s_sentinel1, s_sentinel2):
        for value in coll.values():
            for v in value.values():
                shapes.add(len(v))
    shapes -= {1,}
    if len(shapes) > 1:
        raise Exception(f"Found shapes {shapes} for coordinate {coordinate}")
    print(f"Resulting shapes:", shapes)


def sample_fields(
    coordinates: List[Tuple[float, float]],
    boundaries: List[List[Tuple[float, float]]],
    years: List[int],
    cache_path: Path,
    startdate_postfix: str = "-11-01",
    enddate_postfix: str = "-04-30",
    n_pixels: int = 100,
    overwrite: bool = False,
) -> None:
    """
    Sample all the requested lists, as specified by their boundaries and year.

    :param coordinates: Field coordinates (lat,lng) used to link to correct cache-file
    :param boundaries: Field boundary polygons as a list of (lat,lng) coordinates
    :param years: List of years corresponding each field
    :param cache_path: Directory of data cache
    :param startdate_postfix: Starting date postfix of the previous year (month and day)
    :param enddate_postfix: Ending date postfix of the previous year (month and day)
    :param n_pixels: Number of pixels to sample per time-period (day)
    :param overwrite: Overwrite previously stored data
    """
    for coor, year, boundary in tqdm(
        zip(coordinates, years, boundaries),
        total=len(coordinates),
        desc="Sampling fields...",
    ):
        sample_field(
            coordinate=coor,
            year=year,
            boundary=boundary,
            cache_path=cache_path,
            startdate_postfix=startdate_postfix,
            enddate_postfix=enddate_postfix,
            n_pixels=n_pixels,
            overwrite=overwrite,
        )
