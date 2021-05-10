"""
Export the NDVI, EVI, and NDTI data from the requested fields.

See the compulsatory notebook: tillage_detection.ipynb .
"""
import json
import re
from datetime import datetime
from pathlib import Path
from shutil import rmtree
from time import sleep
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.radix_co2_reduction.earth_engine.datasets import (
    CroplandCollection,
    Landsat7Collection,
    Landsat8Collection,
    Sentinel2Collection,
)
from src.radix_co2_reduction.earth_engine.session import start
from src.radix_co2_reduction.earth_engine.utils import (
    create_bounding_box,
    create_polygon,
    to_polygon,
)

ROOT = Path(__file__).parent / "../../.."

# Load in field-boundaries
with open(Path.home() / "data/agoro/polygons_adj.json", "r") as f:
    boundaries = json.load(f)


def percentile_adjustment(
    data: List[float],
    min_perc: float = 0.25,
    max_perc: float = 0.75,
) -> List[float]:
    """
    Prune the provided data (list of floats) to only contain values [min_perc..max_perc].

    :param data: List of values sampled from the region
    :param min_perc: Minimum percentile, data under this percentile is ignored
    :param max_perc: Maximum percentile, data above this percentile is ignored
    """
    data = sorted(data)
    return data[round(min_perc * len(data)) : round(max_perc * len(data))]


def get_color(coll: str) -> str:
    """Get suiting color for the path."""
    if coll == "landsat7":
        return "tab:blue"
    if coll == "landsat8":
        return "tab:orange"
    if coll == "sentinel2":
        return "tab:green"
    if coll == "cropland":
        return "tab:olive"
    return "black"


def avg(lst: List[float]) -> float:
    """Calculate the average."""
    return sum(lst) / max(len(lst), 1)


def plot_time_series(
    band: str,
    data: List[Tuple[str, Dict[str, List[float]]]],
    write_path: Path,
    comb_f: Callable[..., float] = avg,
    min_perc: float = 0.25,
    max_perc: float = 0.75,
    min_val: float = 0.0,
    colors: Optional[List[str]] = None,
    y_min: float = 0.0,
    y_max: float = 1.0,
    dpi: int = 200,
    plant_date: Optional[str] = None,
    harvest_date: Optional[str] = None,
) -> None:
    """
    Plot a time-series of the band from a collection over a specified region.

    :param band: Band to plot (only used for styling)
    :param data: List of samples (date, list of values) sampled from the region
    :param write_path: Path to write the figure to
    :param comb_f: Function used to combine the samples of a single field into a single representative (float)
    :param min_perc: Minimum percentile, data under this percentile is ignored
    :param max_perc: Maximum percentile, data above this percentile is ignored
    :param min_val: Ignore samples under this value (likely incorrect)
    :param colors: Optionally, list of colors to attach to each data sample
    :param y_min: Minimal value displayed on y-axis
    :param y_max: Maximal value displayed on y-axis
    :param dpi: DPI of the image (to set resolution)
    :param plant_date: Date when seeds planted
    :param harvest_date: Date when harvested
    """
    if not colors:
        colors = [
            "black",
        ] * len(data)

    # Adjust the sampled data and combine
    dates, values = zip(*data)
    values = [[x for x in v[band] if x] for v in values]  # type: ignore
    values = [  # type: ignore
        comb_f(percentile_adjustment(v, min_perc=min_perc, max_perc=max_perc)) for v in values
    ]

    # Filter on value
    if min_val and min(values) < min_val:
        values_temp, dates_temp, colors_temp = [], [], []
        for v, t, c in zip(values, dates, colors):
            if v >= min_val:
                values_temp.append(v)
                dates_temp.append(t)
                colors_temp.append(c)
        values = values_temp  # type: ignore
        dates = dates_temp  # type: ignore
        colors = colors_temp

    # Plot the result
    x_data = np.asarray([pd.to_datetime(d, format="%Y-%m-%d") for d in dates])
    y_data = np.asarray(values)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.scatter(
        x_data,
        y_data,
        label=band,
        c=colors,
    )

    # Draw vline on plant and harvest date, if provided
    if plant_date is not None:
        plt.axvline(
            x=pd.to_datetime(plant_date, format="%Y-%m-%d"),
            color="green",
        )
    if harvest_date is not None:
        plt.axvline(
            x=pd.to_datetime(harvest_date, format="%Y-%m-%d"),
            color="red",
        )

    # Format the date-axis
    fmt_month = mdates.MonthLocator()
    ax.xaxis.set_minor_locator(fmt_month)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    # datemin = np.datetime64(x_data[0], 'Y')
    # datemax = np.datetime64(x_data[-1], 'Y') + np.timedelta64(1, 'Y')
    # ax.set_xlim(datemin, datemax)
    ax.format_xdata = mdates.DateFormatter("%Y-%m")
    fig.autofmt_xdate()
    ax.set_xlabel("Date", fontsize=14)

    # Format the value-axis (y-axis)
    ax.set_ylabel(band, fontsize=14)
    ax.set_ylim(y_min, y_max)

    # Custom legend
    dummy_lines = [
        Line2D([0], [0], color="tab:blue", lw=4),
        Line2D([0], [0], color="tab:orange", lw=4),
        Line2D([0], [0], color="tab:green", lw=4),
        Line2D([0], [0], color="black", lw=4),
        Line2D([0], [0], color="green", lw=4),
        Line2D([0], [0], color="red", lw=4),
    ]
    ax.legend(
        dummy_lines,
        ["Landsat7", "Landsat8", "Sentinel2", "Any", "Plant date", "Harvest date"],
        fontsize=12,
    )

    # Other styling attributes
    ax.set_title(f"{band} over time", fontsize=16)
    ax.grid(lw=0.2)
    plt.tight_layout()
    plt.savefig(write_path, dpi=dpi)
    plt.close()


def extract_harvest_date(d: str) -> Optional[str]:
    """Parse out the harvest date."""
    d = re.sub(r"[\n]+", " ", d)[11:21]
    try:
        return datetime.strptime(d, "%m/%d/%Y").strftime("%Y-%m-%d")
    except ValueError:
        return None


def extract_planting_date(d: str) -> Optional[str]:
    """Parse out the harvest date."""
    d = re.sub(r"[\n]+", " ", d)[9:19]
    try:
        return datetime.strptime(d, "%m/%d/%Y").strftime("%Y-%m-%d")
    except ValueError:
        return None


def export(
    field: Any,
    overwrite: bool = False,
    min_samples: int = 5,
    export_time_series: bool = True,
    debug: bool = False,
) -> bool:
    """
    Export the NDVI, EVI, and NDTI values for the given field.

    :param field: Field to export data from
    :param overwrite: Overwrite previous data stored
    :param min_samples: Minimum number of samples (dates) the field should have
    :param export_time_series: Export time-series images
    :param debug: Export additional information
    """
    ID = int(field.id)
    YEAR = int(field.year)
    FIELD_PATH = Path.home() / f"data/agoro/{datetime.now().strftime('%Y-%m-%d')}/{ID}"
    if not overwrite and (FIELD_PATH / "ndti.png").is_file():
        print(" ! DATA ALREADY EXISTS --> IGNORING")
        return False
    FIELD_PATH.mkdir(parents=True, exist_ok=True)
    TILL_TYPE = field.tillage[9:]
    print(f" - Tillage type: {TILL_TYPE}")

    # Extract harvesting and planting dates
    harvest_date = extract_harvest_date(field.harvest_date)
    print(f" - Harvesting date: {harvest_date}")
    planting_date = extract_planting_date(field.planted_date)
    print(f" - Planting date: {planting_date}")
    startdate = f"{YEAR - 1}-09-01" if debug else f"{YEAR - 1}-11-01"
    enddate = harvest_date if debug else f"{YEAR}-05-31"
    # startdate = f"{YEAR - 1}-06-01" if debug else f"{YEAR - 1}-12-01"
    # enddate = harvest_date if debug else planting_date
    print(f" - Extracting data from {startdate} until {enddate}")

    # Write down the metadata
    with open(FIELD_PATH / "meta.json", "w") as f:
        json.dump(
            {
                "id": ID,
                "year": YEAR,
                "longitude": field.lng,
                "latitude": field.lat,
                "tillage": TILL_TYPE,
                "harvest": harvest_date,
                "planting": planting_date,
                "debug": debug,
                "pdf": f"https://www.beckshybrids.com/Portals/0/SiteContent/YieldData/{YEAR}/{ID}.pdf",
            },
            f,
            indent=2,
        )

    # Create bounding box around field's coordinates
    polygon = to_polygon(
        create_bounding_box(
            lng=field.lng,
            lat=field.lat,
            offset=1000,
        )
    )

    # Load in the field shape file and transform to Earth Engine polygon
    coordinates = boundaries[str(ID)]
    field_polygon = create_polygon([coordinates])
    print(" - Fetched field boundaries")

    # Landsat 7 data: https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LE07_C01_T1_SR
    landsat7 = Landsat7Collection()
    landsat7.load_collection(
        region=field_polygon,
        startdate=startdate,
        enddate=enddate,
        filter_clouds=True,
        filter_perc=0.75,
        relevant_bands=["B1", "B2", "B3", "B4", "B5", "B7", "pixel_qa"],
    )
    if debug:
        print(f" - Number of data samples for {landsat7}: {landsat7.get_size()}")

    # Landsat 8 data: https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C01_T1_SR
    landsat8 = Landsat8Collection()
    landsat8.load_collection(
        region=field_polygon,
        startdate=startdate,
        enddate=enddate,
        filter_clouds=True,
        filter_perc=0.75,
        relevant_bands=["B2", "B3", "B4", "B5", "B6", "B7", "pixel_qa"],
    )
    if debug:
        print(f" - Number of data samples for {landsat8}: {landsat8.get_size()}")

    # Sentinel-2 L2A data: https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR
    sentinel2 = Sentinel2Collection()
    sentinel2.load_collection(
        region=field_polygon,
        startdate=startdate,
        enddate=enddate,
        filter_clouds=True,
        filter_perc=0.75,
        relevant_bands=["B2", "B3", "B4", "B8", "B11", "B12", "QA60", "SCL"],
    )
    if debug:
        print(f" - Number of data samples for {sentinel2}: {sentinel2.get_size()}")

    # Export all true-color images of the datasets
    if debug:
        path = FIELD_PATH / "true_color"
        path.mkdir(parents=True, exist_ok=True)
        landsat7.export_as_png(write_path=path, region=polygon)
        landsat8.export_as_png(write_path=path, region=polygon)
        sentinel2.export_as_png(write_path=path, region=polygon)

    # Add Cropland data layer to all image collections and use it as a mask
    cropland = CroplandCollection()
    cropland.load_collection(
        region=polygon,
        startdate=f"{YEAR}-01-01",
        enddate=f"{YEAR}-12-31",
    )
    cropland_im = cropland.collection.first().select("cropland")
    landsat7.mask_cropland(cropland_im=cropland_im, region=field_polygon)
    size_landsat7 = landsat7.get_size()
    print(f" - Number of data samples for {landsat7} after cropland masking: {size_landsat7}")
    landsat8.mask_cropland(cropland_im=cropland_im, region=field_polygon)
    size_landsat8 = landsat8.get_size()
    print(f" - Number of data samples for {landsat8} after cropland masking: {size_landsat8}")
    sentinel2.mask_cropland(cropland_im=cropland_im, region=field_polygon)
    size_sentinel2 = sentinel2.get_size()
    print(f" - Number of data samples for {sentinel2} after cropland masking: {size_sentinel2}")
    if size_landsat7 + size_landsat8 + size_sentinel2 < min_samples:
        rmtree(FIELD_PATH)
        print(" ! NOT ENOUGH SAMPLES FOUND --> REMOVING FOLDER")
        return False

    # Add the extra layers
    landsat7.add_extra_layers()
    landsat8.add_extra_layers()
    sentinel2.add_extra_layers()

    # Sample the datasets
    backup = FIELD_PATH / "samples"
    backup.mkdir(parents=True, exist_ok=True)
    s_landsat7 = landsat7.sample(
        region=field_polygon,
    )
    with open(backup / "landsat7.json", "w") as f:
        json.dump(s_landsat7, f)
    s_landsat8 = landsat8.sample(
        region=field_polygon,
    )
    with open(backup / "landsat8.json", "w") as f:
        json.dump(s_landsat8, f)
    s_sentinel2 = sentinel2.sample(
        region=field_polygon,
    )
    with open(backup / "sentinel2.json", "w") as f:
        json.dump(s_sentinel2, f)

    # Combine all sampled data by band
    data = list(s_landsat7.items()) + list(s_landsat8.items()) + list(s_sentinel2.items())

    # Check if enough samples exists, ignore field and remove folder if not
    if len(data) < min_samples:
        rmtree(FIELD_PATH)
        print(
            f" ! NOT ENOUGH SAMPLES EXTRACTED "
            f"(Total: {len(data)}, LS7: {len(s_landsat7)}, LS8: {len(s_landsat8)}, S2: {len(s_sentinel2)}) "
            f"--> REMOVING FOLDER"
        )
        return False

    # Plot the time series data
    if export_time_series:
        colors = (
            [
                get_color("landsat7"),
            ]
            * len(s_landsat7)
            + [
                get_color("landsat8"),
            ]
            * len(s_landsat8)
            + [
                get_color("sentinel2"),
            ]
            * len(s_sentinel2)
        )
        plot_time_series(
            band="NDVI",
            data=data,
            colors=colors,
            write_path=FIELD_PATH / "ndvi.png",
            plant_date=planting_date,
            harvest_date=harvest_date if debug else None,
        )
        plot_time_series(
            band="EVI",
            data=data,
            colors=colors,
            write_path=FIELD_PATH / "evi.png",
            plant_date=planting_date,
            harvest_date=harvest_date if debug else None,
        )
        plot_time_series(
            band="NDTI",
            data=data,
            colors=colors,
            write_path=FIELD_PATH / "ndti.png",
            y_max=0.5,  # Same as in indigo paper
            plant_date=planting_date,
            harvest_date=harvest_date if debug else None,
        )
    return True


if __name__ == "__main__":
    # Start an Earth Engine session
    start()

    # Load in all field-data
    beck = pd.read_csv(ROOT / "data/beck_corn_data.csv", index_col=0)

    # Get the fields of interest
    no_till = [
        int(x)
        for x in beck[beck["tillage"] == "TILLAGE: No-Till"]["id"]
        if str(int(x)) in boundaries
    ]
    print(f"Total of {len(no_till)} recognised no-tillage fields")
    conv_till = [
        int(x)
        for x in beck[beck["tillage"] == "TILLAGE: Conv.-Till"]["id"]
        if str(int(x)) in boundaries
    ]
    print(f"Total of {len(conv_till)} recognised conventional tillage fields")

    # Intertwine tillage types
    foi = []
    for n, c in zip(reversed(no_till), reversed(conv_till)):  # Preference for newest fields
        foi += [n, c]
    foi += [x for x in no_till + conv_till if x not in foi]
    print(f"Total of {len(foi)} 'fields of interest'")

    # Iteratively export all the fields
    for i, f in enumerate(foi):  # type: ignore
        print(
            f"\nExporting field with ID '{f}' at {datetime.now()} ({100 * i / len(foi):.2f}% finished)"
        )
        if export(beck[beck["id"] == f].iloc[0], debug=False):
            sleep(5)  # Necessary to prevent download from halting (due to too frequent requests)
