"""Utilisation functions."""
from math import cos, pi
from pathlib import Path
from typing import Any, List, Tuple

import ee
from requests import get


def to_polygon(geojson: Any) -> ee.Geometry:
    """Transform a given geojson to an Earth Engine Polygon."""
    return ee.Geometry(geojson["features"][0]["geometry"])


def create_polygon(coordinates: List[List[Tuple[float, float]]]) -> ee.Geometry:
    """Transform a given geometry to an Earth Engine Polygon."""
    return ee.Geometry(
        {
            "type": "Polygon",
            "coordinates": coordinates,
        }
    )


def get_image_mean(image: ee.Image, region: Any) -> Any:
    """Get the mean value of the image over the specified region."""
    return image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=1,
        maxPixels=1000000000,
    )


def create_bounding_box(
    lng: float,
    lat: float,
    offset: int = 2000,
) -> Any:
    """
    Create a bounding box around the (lon,lat) center with an offset expressed in meters.

    :param lng: Longitude in degrees
    :param lat: Latitude in degrees
    :param offset: Offset in meters, which creates a bounding box of 2*offset by 2*offset
    :return: GeoJSON of a polygon
    """
    dlat, dlng = get_dlat_dlng(
        lat=lat,
        dx=offset,
        dy=offset,
    )
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": [
            {
                "type": "Feature",
                "properties": {"FID": 0},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [lng - dlng, lat - dlat],
                            [lng + dlng, lat - dlat],
                            [lng + dlng, lat + dlat],
                            [lng - dlng, lat + dlat],
                            [lng - dlng, lat - dlat],
                        ]
                    ],
                },
            }
        ],
    }


def get_dlat_dlng(
    lat: float,
    dx: int,
    dy: int,
) -> Tuple[float, float]:
    """
    Get the latitude and longitude relative to the provided coordinates under the offset in meters.

    :param lat: Latitude in degrees
    :param dx: Longitude offset in meters
    :param dy: Latitude offset in meters
    """
    # Express offset in longitude and latitude
    #  https://gis.stackexchange.com/a/2980
    r = 6378137  # Earth radius
    dlat = dy / r * 180 / pi
    dlng = dx / (r * cos(pi * lat / 180)) * 180 / pi
    return dlat, dlng


def download_as_png(
    im: ee.Image,
    vis_param: Any,
    region: Any,
    write_path: Path,
    dimensions: Tuple[int, int] = (512, 512),
) -> None:
    """Download the given image as a PNG."""
    params = dict(vis_param)
    params["region"] = region
    params["dimensions"] = f"{dimensions[0]}x{dimensions[1]}"
    url = im.getThumbURL(params)

    img_data = get(url).content
    with open(write_path, "wb") as handler:
        handler.write(img_data)
