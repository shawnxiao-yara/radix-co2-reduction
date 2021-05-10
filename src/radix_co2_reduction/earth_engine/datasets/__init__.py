"""Google Earth Engine dataset wrappers."""
from .cropland import CroplandCollection
from .landsat7 import Landsat7Collection
from .landsat8 import Landsat8Collection
from .naip import NaipCollection
from .sentinel2 import Sentinel2Collection

__all__ = [
    "CroplandCollection",
    "Landsat7Collection",
    "Landsat8Collection",
    "NaipCollection",
    "Sentinel2Collection",
]
