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


def sar_vv(sample: Dict[str, List[Optional[float]]]) -> List[float]:
    """Get the SAR_VV from the sample."""
    return [x for x in sample["SAR_VV"] if x]


def sar_vh(sample: Dict[str, List[Optional[float]]]) -> List[float]:
    """Get the SAR_VH from the sample."""
    return [x for x in sample["SAR_VH"] if x]
