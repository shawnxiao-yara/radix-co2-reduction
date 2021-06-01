"""Models used for the tillage classification task."""
from src.co2_reduction.tillage_detection.models.field_rf import FieldRF
from src.co2_reduction.tillage_detection.models.pixel_cnn import PixelCNN

__all__ = ["FieldRF", "PixelCNN"]
