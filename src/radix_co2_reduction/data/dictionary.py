"""Dictionary class to capture the data."""
from typing import Any, Dict, Optional


class Data(dict):  # type: ignore
    """Custom data class to store days of field-samples."""

    def __init__(self) -> None:
        """Initialise the data class."""
        super().__init__()

    def update(self, new_dict: Optional[Dict[str, Any]] = None, **F: Any) -> None:  # type: ignore
        """
        D.update([E, ]**F) -> None.  Update D from dict/iterable E and F.

        If E is present and has a .keys() method, then does:  for k in E: D[k] = E[k]
        If E is present and lacks a .keys() method, then does:  for k, v in E: D[k] = v
        In either case, this is followed by: for k in F:  D[k] = F[k]
        """
        if new_dict is None:
            return
        for key, value in new_dict.items():
            self[key] = value

    def __setitem__(self, *args: Any, **kwargs: Any) -> None:
        """Set self[key] to value."""
        key, value = args
        value = Value(value)
        if key in self.keys():
            value.update(self[key])
        super().__setitem__(key, value)


class Value(dict):  # type: ignore
    """Value class which contains values of single day over all supported bands."""

    BANDS = ["R", "G", "B", "NIR", "SWIR1", "SWIR2", "NDVI", "EVI", "NDTI", "SAR_VV", "SAR_VH"]

    def __init__(self, values: Optional[Dict[str, Any]] = None) -> None:
        """Initialise the value class."""
        if values is None:
            values = {}
        for band in self.BANDS:
            if band not in values.keys():
                values[band] = []
        super().__init__(values)

    def update(self, new_dict: Optional[Dict[str, Any]] = None, **F: Any) -> None:  # type: ignore
        """
        D.update([E, ]**F) -> None.  Update D from dict/iterable E and F.

        If E is present and has a .keys() method, then does:  for k in E: D[k] = E[k]
        If E is present and lacks a .keys() method, then does:  for k, v in E: D[k] = v
        In either case, this is followed by: for k in F:  D[k] = F[k]
        """
        if new_dict is None:
            return
        for band in self.BANDS:
            # Ignore if new_dict does not contain band
            if band not in new_dict.keys() or not new_dict[band]:
                continue

            # Replace if no value for band in self
            if not self[band]:
                self[band] = new_dict[band]

            # Merge both
            if len(self[band]) != len(new_dict[band]):
                print(len(self[band]))
                print(len(new_dict[band]))
            assert len(self[band]) == len(new_dict[band])
            self[band] = [_avg(a, b) for a, b in zip(self[band], new_dict[band])]


def _avg(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Calculate the average between two optional floats."""
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return (a + b) / 2
