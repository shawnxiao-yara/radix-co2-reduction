"""Data class for Landsat 8 imagery."""
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import ee

from .base import EarthEngineCollection

NORMALISE_F = {
    "B": (1.0884327621943228, 19.22407971615371),
    "G": (1.0837195311885495, -68.41594723747096),
    "R": (1.0677278435847697, -7.207318548633111),
    "NIR": (0.9753691863053184, 23.108379232653533),
    "SWIR1": (1.0178813893209906, 142.71785656044312),
    "SWIR2": (1.039069455146489, 138.69849745406736),
}


class Landsat8Collection(EarthEngineCollection):
    """Landsat8 Surface Reflectance data collection."""

    def __init__(self) -> None:
        """Initialise the Landsat 8 collection."""
        super().__init__(
            tag="LANDSAT/LC08/C01/T1_SR",
            test_band="B5",  # NIR
        )
        self.band_translation = {
            "R": "B4",
            "G": "B3",
            "B": "B2",
            "NIR": "B5",
            "SWIR1": "B6",
            "SWIR2": "B7",
        }

    def __str__(self) -> str:
        """Representation of the data collection."""
        return "Landsat 8 SR-T1"

    def remove_clouds(
        self,
        region: Any,
        perc: float = 0.25,
        return_masked: bool = True,
    ) -> None:
        """Mask out clouds on the images from the collection."""
        mask_cloud = get_cloud_mask(region, self._test_band, perc=perc, return_masked=return_masked)
        self.collection = self.collection.map(mask_cloud, opt_dropNulls=True)  # type: ignore

    def export_as_png(
        self,
        write_path: Path,
        region: Any,
        vis_params: Optional[Dict[str, Any]] = None,
        postfix: str = "_landsat8",
        dimensions: Tuple[int, int] = (512, 512),
    ) -> None:
        """Export the data collection as PNG images."""
        super().export_as_png(
            write_path=write_path,
            region=region,
            vis_params=vis_params,
            postfix=postfix,
            dimensions=dimensions,
        )

    def set_core_bands(self) -> None:
        """Set the core bands: ['B','G','R','NIR','SWIR1','SWIR2']."""
        for band in ("B", "G", "R", "NIR", "SWIR1", "SWIR2"):

            def _create(im: ee.Image) -> ee.Image:
                """Create the new band."""
                im_band = im.select(self.band_translation[band]).toFloat()
                slope, val = NORMALISE_F[band]
                im_adj = im_band.multiply(slope).add(val)
                im_norm = im_adj.divide(5000.0).max(0.0).min(1.0).toFloat()
                return im.addBands(im_norm.rename(band))

            self.collection = self.collection.map(_create)  # type: ignore


def get_cloud_mask(
    region: Any,
    band: str,
    perc: float = 0.25,
    return_masked: bool = True,
) -> Callable[..., Any]:
    """Create a cloud mask over the specified region."""

    def mask_cloud(image: ee.Image) -> Optional[ee.Image]:
        """Cloud mask over the specified region."""
        # Bits: 3 for cloud shadow, 4 for snow, and 5 for cloud
        cloud_shadow_bit_mask = 1 << 3
        snow_bit_mask = 1 << 4
        cloud_bit_mask = 1 << 5

        # Get the pixel QA band
        qa = image.select("pixel_qa")

        # Flags should be set to zero, indicating clear conditions
        mask = (
            qa.bitwiseAnd(cloud_shadow_bit_mask)
            .eq(0)
            .And(qa.bitwiseAnd(cloud_bit_mask).eq(0))
            .And(qa.bitwiseAnd(snow_bit_mask).eq(0))
        )
        image_masked = image.updateMask(mask)

        # Check if at least perc the image remains, return None if not
        b = image.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=region,
            scale=10,
            maxPixels=1e9,
        )
        a = image_masked.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=region,
            scale=10,
            maxPixels=1e9,
        )
        return ee.Algorithms.If(
            ee.Number(a.get(band)).eq(ee.Number(0)),
            None,
            ee.Algorithms.If(
                ee.Number(b.get(band)).multiply(perc).lte(a.get(band)),
                ee.Algorithms.If(
                    return_masked,
                    image_masked,
                    image,
                ),
                None,
            ),
        )

    return mask_cloud
