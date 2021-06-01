"""Load in the cache."""
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd
from tqdm import tqdm

from src.co2_reduction.data import load_data
from src.co2_reduction.earth_engine.session import start
from src.co2_reduction.utils import sample


def load_beck_tillage() -> Tuple[List[Tuple[float, float]], List[int], List[bool]]:
    """
    Load in the raw Beck's dataset.

    :return: Three lists: (1) the (lat,lng) coordinates, (2) the year, and (3) whether or not there was tillage
    """
    beck = pd.read_csv(Path(__file__).parent / "../../data/beck_corrected.csv")
    coordinates, years, labels = [], [], []
    for _, row in beck.iterrows():
        if row.tillage in ("No-Till", "Conv.-Till"):
            coordinates.append((float(row.lat), float(row.lng)))
            years.append(int(row.year))
            labels.append(row.tillage == "Conv.-Till")

    # Print out overview of the data
    print(f"Loaded in {len(labels)} samples from Beck's dataset:")
    for label in set(labels):
        print(f" - {labels.count(label)} {'Conv.-Till' if label else 'No-Till'}")
    return coordinates, years, labels


def get_features(
    cache_path: Path,
    coordinates: List[Tuple[float, float]],
    model: Any,
) -> List[Any]:
    """
    Get all the features for the given coordinates.

    :param cache_path: Path towards a caching directory to store intermediate results
    :param coordinates: The coordinates to get the features for
    :param model: Model used to create features
    """
    inputs = [(model, cache_path / f"{c[0]}-{c[1]}") for c in coordinates]
    with Pool(cpu_count() - 2) as p:
        features = list(
            tqdm(p.imap(get_single_feature, inputs), total=len(inputs), desc="Creating features...")
        )
    return features


def get_single_feature(inp: Any) -> Any:
    """Get a single coordinate/year couple's features."""
    model, path = inp
    return model.get_features(load_data(path))


def load_cache(
    cache_folder: Path,
    coordinates: List[Tuple[float, float]],
    years: List[int],
    labels: List[bool],
    model_folder: Path = Path(__file__).parent / "../../models",
    min_samples: int = 5,
    n_pixels: int = 100,
) -> None:
    """
    Train the model to predict if a tillage event has occurred on the given coordinate or not.

    :param cache_folder: Path towards a caching directory to store intermediate results (required due to large data size)
    :param coordinates: Field coordinates used to train on
    :param years: Years of the corresponding fields
    :param labels: Tillage labels of the corresponding fields
    :param model_folder: Path towards the directory where the models are stored
    :param min_samples: Minimum number of samples/field required before considering the field for training
    :param n_pixels: Number of pixels to sample for each field
    """
    # Sample the fields
    coordinates, _, labels = sample(
        cache_folder=cache_folder,
        coordinates=coordinates,
        years=years,
        labels=labels,
        model_folder=model_folder,
        min_samples=min_samples,
        n_pixels=n_pixels,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-folder", default=Path.home() / "data/agoro/cache", type=str)
    parser.add_argument("--model-folder", default=Path(__file__).parent / "../../models", type=str)
    parser.add_argument("--n_pixels", default=100, type=int)
    args = parser.parse_args()

    # Setup cache
    cache = Path(args.cache_folder)
    cache.mkdir(parents=True, exist_ok=True)

    # Load in model-path
    models = Path(args.model_folder)

    # Start an Earth Engine session
    start()

    # Load in Beck's dataset
    print("Loading in Beck's dataset...")
    beck_coordinates, beck_years, beck_labels = load_beck_tillage()

    # Load in the samples
    load_cache(
        cache_folder=cache,
        coordinates=beck_coordinates,
        years=beck_years,
        labels=beck_labels,
        model_folder=models,
        n_pixels=args.n_pixels,
    )
