"""Data class for Sentinel-1 imagery."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ee
from tqdm import tqdm

from .base import EarthEngineCollection


class Sentinel1Collection(EarthEngineCollection):
    """Sentinel-1 Synthetic-Aperture Radar (SAR) Ground Range Detected (GRD) data collection."""

    def __init__(self) -> None:
        """Initialise the Sentinel-1 collection."""
        super().__init__(
            tag="COPERNICUS/S1_GRD",
            vis_param={
                "min": 0.0,
                "max": 1.0,
            },
            test_band="VV",
        )

    def __str__(self) -> str:
        """Representation of the data collection."""
        return "Sentinel-1 SAR GRD"

    def load_collection(
        self,
        region: Any,
        startdate: str,
        enddate: str,
        relevant_bands: Optional[List[str]] = None,
        filter_clouds: bool = False,
        filter_perc: float = 0.25,
        merge_same_days: bool = True,
        return_masked: bool = True,
    ) -> None:
        """
        Load in the corresponding data collection.

        :param region: Region that must be present in collection
        :param startdate: Starting date of the collection
        :param enddate: Ending date of the collection
        :param relevant_bands: Only bands to keep (improve performance on large collections)
        :param filter_clouds: Filter the clouds from the collection
        :param filter_perc: Percentage of non-cloud pixels remaining before filtering out image
        :param merge_same_days: Merge images that are taken on the same day
        :param return_masked: Return the image where the clouds are masked out
        """
        self.collection = (
            ee.ImageCollection(self.tag)
            .filterDate(startdate, enddate)
            .filterBounds(region)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(
                ee.Filter.listContains("transmitterReceiverPolarisation", "VV").And(
                    ee.Filter.listContains("transmitterReceiverPolarisation", "VH")
                )
            )
            .filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING"))
            .filter(ee.Filter.eq("resolution", "H"))
        )
        self.remove_none(region=region)
        if merge_same_days:
            self.merge_same_days()
        self.set_core_bands()

    def set_core_bands(self) -> None:
        """Set the core normalised SAR band."""

        def _create(im: ee.Image) -> ee.Image:
            """Create the new band."""
            for band in ("VV", "VH"):
                im_band = im.select(band).toFloat()
                im_adj = im_band.add(50.0)  # -50'..1' --> 0'..51'
                im_norm = im_adj.divide(51.0).max(0.0).min(1.0).toFloat()  # 0'..51' --> 0..1
                im = im.addBands(im_norm.rename(f"SAR_{band}"))
            return im

        self.collection = self.collection.map(_create)  # type: ignore

    def add_to_map(
        self,
        mp: Any,
        scheme: str = "all",
        vis_param: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Add the default collection to the given map."""
        vis_param = dict(self.vis_param) if vis_param is None else vis_param
        vis_param["bands"] = ["SAR_VV"]  # Grey image
        return super().add_to_map(
            mp=mp,
            scheme=scheme,
            vis_param=vis_param,
        )

    def sample(
        self,
        pixels: ee.FeatureCollection,
        seed: int = 42,
    ) -> Dict[str, Dict[str, List[Optional[float]]]]:
        """
        Sample the collection over a specified region.

        :param pixels: The pixels to sample over
        :param seed: Randomisation seed
        :return: For every date a dictionary over all the requested bands containing the sampled values
        """
        # Setup a sampling function
        def _sample(im: ee.Image) -> Any:
            """Sample over the given image."""
            return im.sample(
                region=pixels,
                dropNulls=False,
                scale=1,
                seed=seed,
            )

        # Sample the collection
        coll_sampled = (
            self.collection.select(["SAR_VV", "SAR_VH"]).map(_sample).toList(self.collection.size())  # type: ignore
        )

        # Collect the sampled results
        dates = self.get_dates()
        result: Dict[str, Dict[str, List[Optional[float]]]] = {
            d: {"SAR_VV": [], "SAR_VH": []} for d in dates
        }
        for i, date in enumerate(tqdm(dates, desc=f"Sampling {self}")):
            pixel_features = coll_sampled.get(i).getInfo()["features"]
            for pixel in pixel_features:
                result[date]["SAR_VV"].append(pixel["properties"]["SAR_VV"])
                result[date]["SAR_VH"].append(pixel["properties"]["SAR_VH"])
        return result

    def export_as_png(
        self,
        write_path: Path,
        region: Any,
        vis_params: Optional[Dict[str, Any]] = None,
        postfix: str = "_sentinel1",
        dimensions: Tuple[int, int] = (512, 512),
    ) -> None:
        """Export the data collection as PNG images."""
        vis_params = dict(self.vis_param) if vis_params is None else vis_params
        vis_params["bands"] = ["SAR_VV"]  # Grey image
        super().export_as_png(
            write_path=write_path,
            region=region,
            vis_params=vis_params,
            postfix=postfix,
            dimensions=dimensions,
        )
