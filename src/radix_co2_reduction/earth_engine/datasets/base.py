"""Base model for the Earth Engine data."""
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import ee
from tqdm import tqdm

from src.radix_co2_reduction.earth_engine.utils import download_as_png


class EarthEngineCollection:
    """Collection of data from the Earth Engine."""

    def __init__(
        self,
        tag: str,
        vis_param: Optional[Dict[str, Any]] = None,
        test_band: str = "",
    ) -> None:
        """
        Initialise the collection.

        :param tag: Collection tag used to fetch the collection from the Earth Engine
        :param vis_param: Visualisation parameters specific to the collection
        :param test_band: Band used to determine if pixel values exist
        """
        self._test_band = test_band
        self.collection: Optional[ee.ImageCollection] = None
        self.band_translation: Dict[str, str] = {}
        self.vis_param = (
            {"min": 0.0, "max": 0.5, "bands": ["R", "G", "B"]} if vis_param is None else vis_param
        )
        self.tag = tag

    def __str__(self) -> str:
        """Representation of the data collection."""
        raise NotImplementedError

    def __repr__(self) -> str:
        """Representation of the data collection."""
        return str(self)

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
            ee.ImageCollection(self.tag).filterDate(startdate, enddate).filterBounds(region)
        )
        if relevant_bands:
            self.collection = self.collection.select(relevant_bands)
        if filter_clouds:
            self.remove_clouds(region=region, perc=filter_perc, return_masked=return_masked)
        else:
            self.remove_none(region=region)
        if merge_same_days:
            self.merge_same_days()
        self.set_core_bands()

    def remove_clouds(
        self,
        region: Any,
        perc: float = 0.25,
        return_masked: bool = True,
    ) -> None:
        """Mask out clouds on the images from the collection."""
        pass

    def remove_none(
        self,
        region: Any,
    ) -> None:
        """Remove empty images."""

        def mask(image: Any) -> Any:
            r = image.reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=region,
                scale=30,
                maxPixels=1e9,
            )
            return ee.Algorithms.If(ee.Number(r.get(self._test_band)).eq(ee.Number(0)), None, image)

        self.collection = self.collection.map(mask, opt_dropNulls=True)  # type: ignore

    def merge_same_days(self) -> None:
        """Remove duplicate days from the data collection."""
        # Ignore if collection is non-existing
        size = self.get_size()
        if size == 0:
            return

        # Get dates for images in the collection
        lst = self.collection.toList(self.collection.size())  # type: ignore
        unique_dates = lst.map(lambda im: ee.Image(im).date().format("YYYY-MM-dd")).distinct()

        def merge(date: Any) -> ee.Image:
            date = ee.Date(date)
            im = self.collection.filterDate(date, date.advance(1, "day")).mosaic()  # type: ignore
            return im.set(
                "system:time_start", date.millis(), "system:id", date.format("YYYY-MM-dd")
            )

        self.collection = ee.ImageCollection(unique_dates.map(merge))

    def set_core_bands(self) -> None:
        """Set the core bands of the dataset."""
        pass

    def mask_cropland(
        self,
        cropland_im: Any,
        region: Any,
        perc: float = 0.25,
    ) -> None:
        """Mask out non-crops using the provided cropland image."""

        def mask(image: ee.Image) -> Any:
            image = image.addBands(cropland_im)  # Merge in order to align properly
            image_masked = image.updateMask(
                image.select("cropland").eq(1)
            )  # Only Corn (class=1) accepted

            # Check if at least perc the image remains, return None if not
            b = image.reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=region,
                scale=30,
                maxPixels=1e9,
            )
            a = image_masked.reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=region,
                scale=30,
                maxPixels=1e9,
            )
            return ee.Algorithms.If(
                ee.Number(a.get(self._test_band)).eq(ee.Number(0)),
                None,
                ee.Algorithms.If(
                    ee.Number(b.get(self._test_band)).multiply(perc).lte(a.get(self._test_band)),
                    image_masked,
                    None,
                ),
            )

        self.collection = self.collection.map(mask, opt_dropNulls=True)  # type: ignore

    def add_to_map(
        self,
        mp: Any,
        scheme: str = "all",
        vis_param: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Add the default collection to the given map."""
        assert self.collection is not None
        assert scheme in ("all", "first", "last")
        if vis_param is None:
            vis_param = dict(self.vis_param)
            vis_param["bands"] = ["R", "G", "B"]

        if self.get_size() == 0:
            print(f"No data to show for '{self}'!")
            return mp

        if scheme == "first":
            im = self.collection.first()
            date = self.get_dates()[0]
            mp.add_ee_layer(
                im,
                vis_param,
                f"{self} ({date})",
                True,
            )
        elif scheme == "last":
            im = self.collection.mosaic()
            date = self.get_dates()[-1]
            mp.add_ee_layer(
                im,
                vis_param,
                f"{self} ({date})",
                True,
            )
        else:
            lst = self.collection.select(vis_param["bands"]).toList(self.collection.size())
            dates = self.get_dates(sort=False)
            order = sorted(zip(dates, range(len(dates))))
            for date, i in tqdm(order, desc="Adding images"):
                im = ee.Image(lst.get(i))
                mp.add_ee_layer(
                    im,
                    vis_param,
                    f"{self} ({date})",
                    i == 0,  # First one added
                )
        return mp

    def add_extra_layers(self) -> None:
        """Add all the extra layers."""
        self.add_ndvi()
        self.add_ndti()
        self.add_evi()

    def add_ndvi(self) -> None:
        """Add the NDVI band to the collection."""
        if "NDVI" not in self.get_band_names():
            self.collection = self.collection.map(get_ndvi_band_f())  # type: ignore

    def add_ndti(self) -> None:
        """Add the NDTI band to the collection."""
        if "NDTI" not in self.get_band_names():
            self.collection = self.collection.map(get_ndti_band_f())  # type: ignore

    def add_evi(self) -> None:
        """Add the EVI band to the collection."""
        if "EVI" not in self.get_band_names():
            self.collection = self.collection.map(get_evi_band_f())  # type: ignore

    def get_size(self) -> int:
        """Get the size of the collection."""
        assert self.collection is not None
        return self.collection.size().getInfo()  # type: ignore

    def get_dates(self, sort: bool = True) -> List[str]:
        """Get the dates of the images present in the collection."""
        assert self.collection is not None
        dates = [
            datetime.fromtimestamp(int(x / 1000)).strftime("%Y-%m-%d")
            for x in self.collection.aggregate_array("system:time_start").getInfo()
        ]
        return sorted(dates) if sort else dates

    def get_image_by_date(self, date: str) -> Optional[Any]:
        """Get the image matching the provided date, if exists."""
        dates = self.get_dates()
        if date in dates:
            idx = dates.index(date)
            return ee.Image(self.collection.toList(self.collection.size()).get(idx))  # type: ignore
        return None

    def get_band_names(self) -> Any:
        """Get a list of all band-names supported by the collection."""
        if self.get_size():  # Must have an image
            return self.collection.first().bandNames().getInfo()  # type: ignore
        return []

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
        # Make sure all bands exist
        if "NDVI" not in self.get_band_names():
            self.add_extra_layers()
        bands = ["B", "G", "R", "NIR", "SWIR1", "SWIR2", "NDVI", "EVI", "NDTI"]

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
        coll_sampled = self.collection.select(bands).map(_sample).toList(self.collection.size())  # type: ignore

        # Collect the sampled results
        dates = self.get_dates()
        result: Dict[str, Dict[str, List[Optional[float]]]] = {
            d: {b: [] for b in bands} for d in dates
        }
        for i, date in enumerate(tqdm(dates, desc=f"Sampling {self}")):
            pixel_features = coll_sampled.get(i).getInfo()["features"]
            for pixel in pixel_features:
                for band in bands:
                    result[date][band].append(pixel["properties"][band])
        return result

    def export_as_png(
        self,
        write_path: Path,
        region: Any,
        vis_params: Optional[Dict[str, Any]] = None,
        postfix: str = "",
        dimensions: Tuple[int, int] = (512, 512),
    ) -> None:
        """Export the data collection as PNG images."""
        if vis_params is None:
            vis_params = dict(self.vis_param)
            if "palette" in vis_params:
                del vis_params["palette"]
            vis_params["bands"] = ["R", "G", "B"]
        lst = self.collection.select(vis_params["bands"]).toList(self.collection.size())  # type: ignore
        for i in tqdm(range(self.get_size()), desc=f"Exporting {self}"):
            im = ee.Image(lst.get(i))
            date = datetime.fromtimestamp(int(im.date().getInfo()["value"] / 1000)).strftime(
                "%Y-%m-%d"
            )
            download_as_png(
                im=im,
                vis_param=vis_params,
                region=region,
                write_path=write_path / f"{date}{postfix}.png",
                dimensions=dimensions,
            )


def get_pixel_count(image: ee.Image, region: Any) -> Any:
    """Get the number of pixels present in the image over the specified region."""
    return image.reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=region,
        scale=10,
        maxPixels=1e9,
    )


def get_ndvi_band_f(tag: str = "NDVI") -> Callable[..., Any]:
    """Create the function to add the NDVI band to the collection."""

    def add_ndvi_band(image: ee.Image) -> ee.Image:
        """Add the NDVI (Normalised Difference Vegetation Index) band to the given image."""
        return image.addBands(image.normalizedDifference(["NIR", "R"]).rename(tag))

    return add_ndvi_band


def get_ndti_band_f(tag: str = "NDTI") -> Callable[..., Any]:
    """Create the function to add the NDTI band to the collection."""

    def add_ndti_band(image: ee.Image) -> ee.Image:
        """Add the NDTI (Normalised Difference Tillage Index) band to the given image."""
        return image.addBands(image.normalizedDifference(["SWIR1", "SWIR2"]).rename(tag))

    return add_ndti_band


def get_evi_band_f(tag: str = "EVI") -> Callable[..., Any]:
    """Create the function to add the EVI band to the collection."""

    def add_evi_band(image: ee.Image) -> ee.Image:
        """Add the EVI (Enhanced Vegetation Index) band to the given image."""
        nir_b = image.select("NIR")
        r_b = image.select("R")
        b_b = image.select("B")
        num = nir_b.subtract(r_b)
        den = nir_b.add(r_b.multiply(6)).subtract(b_b.multiply(7.5)).add(1)
        return image.addBands(num.divide(den).rename(tag))

    return add_evi_band
