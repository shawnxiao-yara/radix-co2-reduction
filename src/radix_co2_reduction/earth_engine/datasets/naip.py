"""Data class for National Agriculture Imagery Program (NAIP) imagery."""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .base import EarthEngineCollection


class NaipCollection(EarthEngineCollection):
    """National Agriculture Imagery Program data collection."""

    def __init__(self) -> None:
        """Initialise the National Agriculture Imagery Program (NAIP) collection."""
        super().__init__(
            vis_param={
                "min": 0.0,
                "max": 255.0,
            },
            tag="USDA/NAIP/DOQQ",
            test_band="N",
        )

    def __str__(self) -> str:
        """Representation of the data collection."""
        return "NAIP"

    def export_as_png(
        self,
        write_path: Path,
        region: Any,
        vis_params: Optional[Dict[str, Any]] = None,
        postfix: str = "_naip",
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
