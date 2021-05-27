"""Main script (pipeline) for the tillage detection task."""
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm

from src.radix_co2_reduction.data import load_data
from src.radix_co2_reduction.earth_engine.session import start
from src.radix_co2_reduction.field_detect import extract_field_boundaries
from src.radix_co2_reduction.gee_extract import sample_fields
from src.radix_co2_reduction.tillage_detection import FieldRF


def load_beck() -> Tuple[List[Tuple[float, float]], List[int], List[bool]]:
    """
    Load in the raw Beck's dataset.

    :return: Two lists: (1) the (lat,lng) coordinates, and (2) whether or not there was tillage
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


def sample(
    cache_path: Path,
    coordinates: List[Tuple[float, float]],
    years: List[int],
    labels: List[bool],
    model_path: Path = Path(__file__).parent / "../../models",
    min_samples: int = 5,
) -> Tuple[List[Tuple[float, float]], List[bool]]:
    """
    Sample the fields under the given coordinates.

    :param cache_path: Path towards a caching directory to store intermediate results (required due to large data size)
    :param coordinates: Coordinates (lat,lng) of the fields to sample
    :param years: Years in which to sample the corresponding fields
    :param labels: Tillage labels of the corresponding fields
    :param model_path: Path towards the directory where the models are stored
    :param min_samples: Minimum number of samples/field required before considering the field for training
    :return: Tuple of successfully sampled coordinates with their corresponding label
    """
    # Load in all the field boundaries and filter out the samples that aren't recognised
    print("\nPredicting field boundaries...")
    boundaries = extract_field_boundaries(
        coordinates=coordinates,
        model_path=model_path,
        cache_path=cache_path,
    )
    assert len(boundaries) == len(coordinates)
    coordinates_f, years_f, labels_f, boundaries_f = [], [], [], []
    for c, y, l, b in zip(coordinates, years, labels, boundaries):
        if b:
            coordinates_f.append(c)
            years_f.append(y)
            labels_f.append(l)
            boundaries_f.append(b)
    coordinates, years, labels, boundaries = coordinates_f, years_f, labels_f, boundaries_f
    print(f"Total of {len(labels)} field boundaries successfully extracted")

    # Sample fields using Google Earth Engine
    print("\nSampling fields...")
    sample_fields(
        coordinates=coordinates,
        boundaries=boundaries,
        years=years,
        cache_path=cache_path,
    )
    coordinates_f, years_f, labels_f, boundaries_f = [], [], [], []
    for c, y, l, b in zip(coordinates, years, labels, boundaries):  # noqa B007
        if len(load_data(cache_path / f"{c[0]}-{c[1]}")) >= min_samples:
            coordinates_f.append(c)
            labels_f.append(l)
    coordinates, labels = coordinates_f, labels_f
    print(f"Total of {len(labels)} fields successfully sampled")
    return coordinates, labels


def train(
    cache_path: Path,
    coordinates: List[Tuple[float, float]],
    years: List[int],
    labels: List[bool],
    model_path: Path = Path(__file__).parent / "../../models",
    min_samples: int = 5,
) -> None:
    """
    Train the model to predict if a tillage event has occurred on the given coordinate or not.

    :param cache_path: Path towards a caching directory to store intermediate results (required due to large data size)
    :param coordinates: Field coordinates used to train on
    :param years: Years of the corresponding fields
    :param labels: Tillage labels of the corresponding fields
    :param model_path: Path towards the directory where the models are stored
    :param min_samples: Minimum number of samples/field required before considering the field for training
    """
    # Sample the fields
    coordinates, labels = sample(
        cache_path=cache_path,
        coordinates=coordinates,
        years=years,
        labels=labels,
        model_path=model_path,
        min_samples=min_samples,
    )

    # Train the classification model's feature mask first
    print("\nTraining the model...")
    model = FieldRF(
        models_path=model_path,
    )
    model.init_feature_mask()
    model.optimise_feature_mask(
        features=[
            model.get_features(load_data(cache_path / f"{c[0]}-{c[1]}"))
            for c in tqdm(coordinates, desc="Creating features...")
        ],
        labels=labels,
    )

    # Train the classification model itself and save the results
    model.train(
        features=[
            model.get_features(load_data(cache_path / f"{c[0]}-{c[1]}"))
            for c in tqdm(coordinates, desc="Creating features...")
        ],
        labels=labels,
    )
    model.save()


def test(
    cache_path: Path,
    coordinates: List[Tuple[float, float]],
    years: List[int],
    labels: List[bool],
    model_path: Path = Path(__file__).parent / "../../models",
    min_samples: int = 5,
) -> None:
    """
    Evaluate the model's performance on the given coordinates.

    :param cache_path: Path towards a caching directory to store intermediate results (required due to large data size)
    :param coordinates: Field coordinates used to train on
    :param years: Years of the corresponding fields
    :param labels: Tillage labels of the corresponding fields
    :param model_path: Path towards the directory where the models are stored
    :param min_samples: Minimum number of samples/field required before considering the field
    """
    # Sample the fields
    coordinates, labels = sample(
        cache_path=cache_path,
        coordinates=coordinates,
        years=years,
        labels=labels,
        model_path=model_path,
        min_samples=min_samples,
    )

    # Evaluate the classification model's performance
    print("\nEvaluating the model...")
    model = FieldRF(
        models_path=model_path,
    )
    model.eval(
        features=[
            model.get_features(load_data(cache_path / f"{c[0]}-{c[1]}"))
            for c in tqdm(coordinates, desc="Creating features...")
        ],
        labels=labels,
    )


def infer(
    cache_path: Path,
    coordinates: List[Tuple[float, float]],
    years: List[int],
    model_path: Path = Path(__file__).parent / "../../models",
    min_samples: int = 5,
) -> List[bool]:
    """
    Perform inference on the model.

    :param cache_path: Path towards a caching directory to store intermediate results (required due to large data size)
    :param coordinates: Field coordinates used to train on
    :param years: Years of the corresponding fields
    :param model_path: Path towards the directory where the models are stored
    :param min_samples: Minimum number of samples/field required before considering the field
    """
    # Sample the fields
    coordinates, _ = sample(
        cache_path=cache_path,
        coordinates=coordinates,
        years=years,
        labels=[False] * len(coordinates),  # Dummy labels
        model_path=model_path,
        min_samples=min_samples,
    )

    # Load in the model
    print("\nPredicting tillage events...")
    model = FieldRF(
        models_path=model_path,
    )
    return [
        model(load_data(cache_path / f"{c[0]}-{c[1]}"))
        for c in tqdm(coordinates, desc="Predicting tillage events...")
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", default=Path.home() / "data/agoro/cache", type=str)
    parser.add_argument("--model-path", default=Path(__file__).parent / "../../models", type=str)
    parser.add_argument("--train", default=0, type=int)
    parser.add_argument("--test", default=0, type=int)
    parser.add_argument("--infer", default=0, type=int)
    args = parser.parse_args()

    # Setup cache
    cache = Path(args.cache_path)
    cache.mkdir(parents=True, exist_ok=True)

    # Load in model-path
    models = Path(args.model_path)

    # Start an Earth Engine session
    start()

    # Load in Beck's dataset
    print("Loading in Beck's dataset...")
    beck_coordinates, beck_years, beck_labels = load_beck()

    # Train the model, if requested
    if args.train:
        train(
            cache_path=cache,
            model_path=models,
            coordinates=beck_coordinates,
            years=beck_years,
            labels=beck_labels,
        )

    # Test the model, if requested
    if args.test:
        test(
            cache_path=cache,
            model_path=models,
            coordinates=beck_coordinates,
            years=beck_years,
            labels=beck_labels,
        )

    # Perform inference on the model, if requested
    if args.infer:
        preds = infer(
            cache_path=cache,
            model_path=models,
            coordinates=beck_coordinates,
            years=beck_years,
        )
        print(f"Predicted labels: {preds}")
