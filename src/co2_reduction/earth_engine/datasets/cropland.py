"""Data class for Landsat 7 imagery."""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .base import EarthEngineCollection


class CroplandCollection(EarthEngineCollection):
    """USDA NASS Cropland data collection."""

    def __init__(self) -> None:
        """Initialise the Cropland USDA NASS collection."""
        super().__init__(
            tag="USDA/NASS/CDL",
            vis_param={
                "type": "PixelType",
                "precision": "int",
                "min": 1,
                "max": 2,
                "palette": ["yellow", "000000"],
            },
            test_band="cropland",  # Only used for null-detection
        )

    def __str__(self) -> str:
        """Representation of the data collection."""
        return "Cropland USDA NASS"

    def add_to_map(
        self,
        mp: Any,
        scheme: str = "all",
        vis_param: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Add the default collection to the given map."""
        if vis_param is None:
            vis_param = dict(self.vis_param)
        vis_param["bands"] = ["cropland"]
        return super().add_to_map(mp=mp, scheme=scheme, vis_param=vis_param)

    def export_as_png(
        self,
        write_path: Path,
        region: Any,
        vis_params: Optional[Dict[str, Any]] = None,
        postfix: str = "_cropland",
        dimensions: Tuple[int, int] = (512, 512),
    ) -> None:
        """Export the data collection as PNG images."""
        if vis_params is None:
            vis_params = dict(self.vis_param)
            if "palette" in vis_params:
                del vis_params["palette"]
            vis_params["bands"] = ["cropland"]  # Only interested in cropland
        super().export_as_png(
            write_path=write_path,
            region=region,
            vis_params=vis_params,
            postfix=postfix,
            dimensions=dimensions,
        )
