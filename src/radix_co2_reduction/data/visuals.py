"""Create visualisations."""
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from .data import BANDS, get_year, load_data, load_pixel_data
from .utils import datetime_to_int


def heatmap_field(
    field_path: Path,
    pixel_idx: int = 0,
    downsample: int = 10,
) -> None:
    """Heatmap visualisation of a pixel from the requested field."""
    # Load the data, if exists
    data, till = load_pixel_data(
        load_data(field_path),
        year=get_year(field_path),
        downsample=downsample,
    )
    if not data:
        print("No data found")
        return

    # Plot the requested pixel
    sample = data[pixel_idx]
    plt.figure(figsize=(sample.shape[1] // 2, 4))
    plt.title(f"{field_path.name} - {'Conv.-Till' if till else 'No-Till'}")
    sns.heatmap(sample)
    plt.xlabel("Time (old to new)")
    plt.xticks([])
    plt.yticks([i + 0.5 for i in range(len(BANDS))], BANDS, rotation=0)
    plt.show()


def plot_time_series(
    features:Any,
    label:bool,
    pixel_idx: int = 0,
    band: str = "NDVI",
    downsample: int = 10,
    y_max: float = 1.0,
) -> None:
    """Plot the given time-series."""
    assert band in BANDS

    # Load the data, if exists
    data, till = load_pixel_data(
        sample=load_data(field_path),
        year=get_year(field_path),
        downsample=downsample,
    )
    if not data:
        print("No data found")
        return
    dates, raw_values = zip(*sorted(load_data(field_path).items()))
    start_idx = datetime_to_int(f"{get_year(field_path) - 1}-11-01")
    date_idx = [(datetime_to_int(d) - start_idx) / downsample for d in dates]

    # Plot the requested pixel
    sample = data[pixel_idx]
    values = sample[BANDS.index(band)]
    plt.figure(figsize=(sample.shape[1] // 3, 3))
    plt.plot(values)

    # Plot the scattered data
    for date, raw_value in zip(date_idx, raw_values):
        values = [v for v in raw_value[band] if v]
        plt.scatter(date, sum(values) / len(values), color="red", linewidths=0.2)

    # Add general plot attributes
    plt.title(f"{field_path.name} - {'Conv.-Till' if till else 'No-Till'} - {band}")
    plt.xlabel("Time (old to new)")
    plt.xticks([])
    plt.ylim(0, y_max)
    plt.tight_layout()
    plt.show()
