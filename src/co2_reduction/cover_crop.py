"""Main script (pipeline) for the tillage detection task."""
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hdbscan
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from src.co2_reduction.cover_crop_detection import FieldSVM, ndvi_feature
from src.co2_reduction.data import load_data
from src.co2_reduction.earth_engine.session import start
from src.co2_reduction.utils import sample


def load_beck_cover_crop() -> Tuple[List[Tuple[float, float]], List[int], List[Optional[bool]]]:
    """
    Load in the raw Beck's dataset.

    :return: Three lists: (1) the (lat,lng) coordinates, (2) the year, and (3) whether or not there was cover crop
    """
    # Load in Beck's data
    beck = pd.read_csv(Path(__file__).parent / "../../data/beck_corrected.csv")
    beck = beck.where(pd.notnull(beck), None)
    lat, lng = list(beck.lat), list(beck.lng)
    coordinates = list(zip(lat, lng))
    years = list(beck.year)
    labels = list(beck.cover_crop)

    # Print out overview of the data
    print(f"Loaded in {len(labels)} samples from Beck's dataset:")
    for label in set(labels):
        print(f" - {labels.count(label)} {label}")
    return coordinates, years, labels


def get_features(
    cache_folder: Path,
    coordinates: List[Tuple[float, float]],
    years: List[int],
) -> List[Any]:
    """
    Get all the features for the given coordinates.

    :param cache_folder: Path towards a caching directory to store intermediate results
    :param coordinates: The coordinates to get the features for
    :param years: Years specifying the time period to sample the corresponding coordinate
    """
    inputs = [(cache_folder / f"{c[0]}-{c[1]}", y) for c, y in zip(coordinates, years)]
    with Pool(cpu_count() - 2) as p:
        features = list(
            tqdm(p.imap(get_single_feature, inputs), total=len(inputs), desc="Creating features...")
        )
    return features


def get_single_feature(inp: Any) -> Any:
    """Get a single coordinate/year couple's features."""
    p, y = inp
    return ndvi_feature(load_data(p), y)


def train(
    cache_folder: Path,
    coordinates: List[Tuple[float, float]],
    years: List[int],
    labels: List[Optional[bool]],
    model_folder: Path = Path(__file__).parent / "../../models",
    min_samples: int = 5,
) -> None:
    """
    Train the model to predict if a tillage event has occurred on the given coordinate or not.

    :param cache_folder: Path towards a caching directory to store intermediate results (required due to large data size)
    :param coordinates: Field coordinates used to train on
    :param years: Years of the corresponding fields
    :param labels: Tillage labels of the corresponding fields
    :param model_folder: Path towards the directory where the models are stored
    :param min_samples: Minimum number of samples/field required before considering the field for training
    """
    # Sample the fields
    coordinates, years, labels = sample(
        cache_folder=cache_folder,
        coordinates=coordinates,
        years=years,
        labels=labels,
        model_folder=model_folder,
        min_samples=min_samples,
    )

    # Predict None-labels
    features = get_features(cache_folder, coordinates, years)

    # HDBSCAN - Create clusters of similar features
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5,  # min #samples / cluster
        min_samples=2,  # How conservative to cluster
    )
    clusterer.fit(features)
    cluster_val: Dict[int, Optional[bool]] = {-1: None}
    for c_id in set(clusterer.labels_) - {-1}:
        cluster_labels = {
            l for l, c in zip(labels, clusterer.labels_) if c == c_id and l is not None  # noqa E741
        }
        if len(cluster_labels) == 1:
            cluster_val[c_id] = True if True in cluster_labels else False
        else:
            cluster_val[c_id] = None  # Conflicting or unknown cluster
    for i, (l, c) in enumerate(zip(labels, clusterer.labels_)):
        if l is None and cluster_val[c] is not None:
            labels[i] = cluster_val[c]

    # K-NN to label remaining None-labels
    features_labeled, labels_labeled = zip(
        *[(f, l) for f, l in zip(features, labels) if l is not None]
    )
    neighbours = KNeighborsClassifier(
        n_neighbors=1,
    ).fit(features_labeled, labels_labeled)
    labels = neighbours.predict(features)

    # Train the classification model itself and save the results
    print("\nTraining the model...")
    model = FieldSVM(
        model_folder=model_folder,
    )
    model.train(
        features=features,
        labels=labels,
    )
    model.save()


def evaluate(
    cache_folder: Path,
    coordinates: List[Tuple[float, float]],
    years: List[int],
    labels: List[Optional[bool]],
    model_folder: Path = Path(__file__).parent / "../../models",
    min_samples: int = 5,
) -> None:
    """
    Evaluate the model's performance on the given coordinates.

    :param cache_folder: Path towards a caching directory to store intermediate results (required due to large data size)
    :param coordinates: Field coordinates used to train on
    :param years: Years of the corresponding fields
    :param labels: Tillage labels of the corresponding fields
    :param model_folder: Path towards the directory where the models are stored
    :param min_samples: Minimum number of samples/field required before considering the field
    """
    # Only use known labels
    coordinates_f, years_f, labels_f = [], [], []
    for c, y, l in zip(coordinates, years, labels):
        if l is not None:
            coordinates_f.append(c)
            years_f.append(y)
            labels_f.append(l)
    coordinates, years, labels = coordinates_f, years_f, labels_f  # type: ignore

    # Sample the fields
    coordinates, years, labels = sample(
        cache_folder=cache_folder,
        coordinates=coordinates,
        years=years,
        labels=labels,
        model_folder=model_folder,
        min_samples=min_samples,
    )

    # Evaluate the classification model's performance
    print("\nEvaluating the model...")
    model = FieldSVM(
        model_folder=model_folder,
    )
    model.eval(
        features=get_features(cache_folder, coordinates, years),
        labels=labels,
    )


def infer(
    cache_folder: Path,
    coordinates: List[Tuple[float, float]],
    years: List[int],
    model_folder: Path = Path(__file__).parent / "../../models",
    min_samples: int = 5,
) -> List[bool]:
    """
    Perform inference on the model.

    :param cache_folder: Path towards a caching directory to store intermediate results (required due to large data size)
    :param coordinates: Field coordinates used to train on
    :param years: Years of the corresponding fields
    :param model_folder: Path towards the directory where the models are stored
    :param min_samples: Minimum number of samples/field required before considering the field
    """
    # Sample the fields
    coordinates, years, _ = sample(
        cache_folder=cache_folder,
        coordinates=coordinates,
        years=years,
        labels=[False] * len(coordinates),  # Dummy labels
        model_folder=model_folder,
        min_samples=min_samples,
    )

    # Load in the model
    print("\nPredicting tillage events...")
    model = FieldSVM(
        model_folder=model_folder,
    )
    return [
        model(load_data(cache_folder / f"{c[0]}-{c[1]}"), y)
        for c, y in tqdm(
            zip(coordinates, years), total=len(years), desc="Predicting tillage events..."
        )
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-folder", default=Path.home() / "data/agoro/cache", type=str)
    parser.add_argument("--model-folder", default=Path(__file__).parent / "../../models", type=str)
    parser.add_argument("--train", default=1, type=int)
    parser.add_argument("--test", default=1, type=int)
    parser.add_argument("--infer", default=0, type=int)
    args = parser.parse_args()

    # Setup cache
    cache = Path(args.cache_folder)
    cache.mkdir(parents=True, exist_ok=True)

    # Load in model-folder
    models = Path(args.model_folder)

    # Start an Earth Engine session
    start()

    # Load in Beck's dataset
    print("Loading in Beck's dataset...")
    beck_coordinates, beck_years, beck_labels = load_beck_cover_crop()

    # Train the model, if requested
    if args.train:
        train(
            cache_folder=cache,
            model_folder=models,
            coordinates=beck_coordinates,
            years=beck_years,
            labels=beck_labels,
        )

    # Test the model, if requested
    if args.test:
        evaluate(
            cache_folder=cache,
            model_folder=models,
            coordinates=beck_coordinates,
            years=beck_years,
            labels=beck_labels,
        )

    # Perform inference on the model, if requested
    if args.infer:
        preds = infer(
            cache_folder=cache,
            model_folder=models,
            coordinates=beck_coordinates,
            years=beck_years,
        )
        print(f"Predicted labels: {preds}")
