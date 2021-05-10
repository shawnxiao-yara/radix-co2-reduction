"""Method to calculate indices from the sampled bands."""
from typing import Dict, List, Optional


def ndvi(sample: Dict[str, List[Optional[float]]]) -> List[float]:
    """Get the NDVI from the sample."""
    return [x for x in sample["NDVI"] if x]


def evi(sample: Dict[str, List[Optional[float]]]) -> List[float]:
    """Get the EVI from the sample."""
    return [x for x in sample["EVI"] if x]


def ndti(sample: Dict[str, List[Optional[float]]]) -> List[float]:
    """Get the NDTI from the sample."""
    return [x for x in sample["NDTI"] if x]


def r(sample: Dict[str, List[Optional[float]]]) -> List[float]:
    """Get the R from the sample."""
    return [x for x in sample["R"] if x]


def g(sample: Dict[str, List[Optional[float]]]) -> List[float]:
    """Get the G from the sample."""
    return [x for x in sample["G"] if x]


def b(sample: Dict[str, List[Optional[float]]]) -> List[float]:
    """Get the b from the sample."""
    return [x for x in sample["B"] if x]


def nir(sample: Dict[str, List[Optional[float]]]) -> List[float]:
    """Get the NIR from the sample."""
    return [x for x in sample["NIR"] if x]


def swir1(sample: Dict[str, List[Optional[float]]]) -> List[float]:
    """Get the SWIR1 from the sample."""
    return [x for x in sample["SWIR1"] if x]


def swir2(sample: Dict[str, List[Optional[float]]]) -> List[float]:
    """Get the SWIR2 from the sample."""
    return [x for x in sample["SWIR2"] if x]


def r_swir1(sample: Dict[str, List[Optional[float]]]) -> List[float]:
    """Get the index obtained by calculating the normalised difference between SWIR1 and R from the sample."""
    values = [
        (b - a) / ((b + a) if (b + a) != 0 else 1) for a, b in zip(nir(sample), swir1(sample))
    ]
    return values


def r_swir2(sample: Dict[str, List[Optional[float]]]) -> List[float]:
    """Get the index obtained by calculating the normalised difference between SWIR2 and R from the sample."""
    values = [(b - a) / ((b + a) if (b + a) != 0 else 1) for a, b in zip(r(sample), swir2(sample))]
    return values


def nir_swir1(sample: Dict[str, List[Optional[float]]]) -> List[float]:
    """Get the index obtained by calculating the normalised difference between SWIR1 and NIR from the sample."""
    values = [
        (b - a) / ((b + a) if (b + a) != 0 else 1) for a, b in zip(nir(sample), swir1(sample))
    ]
    return values


def nir_swir2(sample: Dict[str, List[Optional[float]]]) -> List[float]:
    """Get the index obtained by calculating the normalised difference between SWIR2 and NIR from the sample."""
    values = [
        (b - a) / ((b + a) if (b + a) != 0 else 1) for a, b in zip(nir(sample), swir2(sample))
    ]
    return values
