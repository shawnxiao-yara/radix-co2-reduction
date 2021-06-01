"""Generate data to annotate fields with."""
from datetime import datetime
from pathlib import Path
from shutil import rmtree
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.co2_reduction.earth_engine.datasets import CroplandCollection, NaipCollection
from src.co2_reduction.earth_engine.session import start
from src.co2_reduction.earth_engine.utils import create_bounding_box, download_as_png, to_polygon

ROOT = Path(__file__).parent / "../.."
WRITE_PATH = Path.home() / "data/agoro/fields_raw"


def generate(
    field: Any,
    incl_cropland: bool = True,
    clean: bool = False,
) -> None:
    """
    Generate the data corresponding the list of field IDs.

    :param field: Field to export
    :param incl_cropland: Include the cropland data layer in the output
    :param clean: Clean folder if already exported, skip otherwise
    """
    path = WRITE_PATH / f"{int(field.id)}"
    if (path / "true_color.png").is_file():
        if clean:
            rmtree(path)
        else:
            return
    path.mkdir(exist_ok=True, parents=True)

    # Create bounding box
    bounding_box = create_bounding_box(
        lng=field.lng,
        lat=field.lat,
        offset=1000,
    )
    polygon = to_polygon(bounding_box)

    # Load in true-color dataset
    year = int(field.year)
    true_color = NaipCollection()
    true_color.load_collection(
        region=polygon,
        startdate="2017-01-01",  # Assumption: Field parcels remain the same over time
        enddate="2020-12-31",
        return_masked=False,  # Only interested in raw images
    )

    # Use latest possible image
    im = true_color.collection.mosaic()

    # Export as PNG
    vis_s = dict(true_color.vis_param)
    vis_s["bands"] = [true_color._r, true_color._g, true_color._b]
    download_as_png(
        im=im,
        vis_param=vis_s,
        region=polygon,
        write_path=path / "true_color.png",
        dimensions=(1000, 1000),
    )

    # Load in cropland dataset and export as PNG, if requested
    if incl_cropland:
        cropland = CroplandCollection()
        cropland.load_collection(
            region=polygon,
            startdate=f"{year}-01-01",
            enddate=f"{year}-12-31",
        )

        # Merge cropland info into the true-color collection to form an image
        im = im.addBands(cropland.collection.first().select("cropland"))

        # Export as PNG
        vis_cl = dict(cropland.vis_param)
        del vis_cl["palette"]
        vis_cl["bands"] = ["cropland"]
        download_as_png(
            im=im,
            vis_param=vis_cl,
            region=polygon,
            write_path=path / "cropland.png",
            dimensions=(1000, 1000),
        )

        # Merge the two images
        im_tc = Image.open(open(str(path / "true_color.png"), "rb"))
        im_tc = im_tc.convert("RGBA")
        im_field = Image.open(open(str(path / "cropland.png"), "rb"))
        im_field = im_field.convert("RGBA")
        arr = np.array(im_field)
        arr[:, :, 0] = 255
        arr[:, :, 3] = np.divide(arr[:, :, 3], 10)
        im = Image.fromarray(arr)
        overlay = Image.alpha_composite(im_tc, im)
        overlay.save(path / "overlay.png")


if __name__ == "__main__":
    # Start an Earth Engine session
    start()

    # Load in all field-data
    beck = pd.read_csv(ROOT / "data/beck_corrected.csv", index_col=0)
    beck.head()

    # Generate the fields
    field_ids = list(reversed([int(x) for x in beck["id"]]))
    pbar = tqdm(total=len(field_ids))
    try:
        for f_id in field_ids:
            pbar.set_description(f"Generating '{f_id}' ({datetime.now().strftime('%H:%M:%S')})")
            generate(
                field=beck[beck["id"] == f_id].iloc[0],
                incl_cropland=False,  # TODO: Currently not needed, makes export faster
                clean=False,
            )
            pbar.update()
    finally:
        pbar.close()
