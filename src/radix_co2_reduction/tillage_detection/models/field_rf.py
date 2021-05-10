"""Random Forest classifier that predicts on field-level."""
import json
import pickle  # noqa S403
from multiprocessing import Pool, cpu_count
from pathlib import Path
from random import shuffle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, f1_score, recall_score
from tqdm import tqdm

from radix_co2_reduction.earth_engine.cloud_filter import CloudFilter
from src.radix_co2_reduction.tillage_detection.features import custom
from src.radix_co2_reduction.tillage_detection.index import ndti, ndvi


class FieldRF:
    """Random Forest classifier that predicts on field-level."""

    def __init__(
        self,
        models_path: Path,
    ) -> None:
        """
        Initialise the classifier.

        :param models_path: Path where models (classification, cloud filter) are stored
        """
        self.models_path = models_path
        self.feature_mask: Optional[List[bool]] = None
        self.clf: Optional[RandomForestClassifier] = None
        self.cloud_filter: Optional[CloudFilter] = None
        self.load()

    def __call__(self, field_sample: Dict[str, Dict[str, List[Optional[float]]]]) -> Any:
        """Predict if sample is tilled."""
        assert self.clf is not None
        assert self.feature_mask is not None

        # Unfold the data, combine all values over time
        data: Dict[str, List[float]] = {}
        for sample in tqdm(field_sample.values(), "Unfolding data"):
            if self.cloud_filter is not None and self.cloud_filter(sample):
                continue
            for band, values in sample.items():
                if band not in data:
                    data[band] = []
                data[band] += values

        # Extract features and make a prediction
        feature = self._get_features(data)
        return self.clf.predict([feature])[0]

    def _get_features(self, sample: Dict[str, List[float]]) -> Any:
        """Get the requested features from the given sample."""
        assert self.feature_mask is not None
        s_feature = []
        for idx_f in (ndvi, ndti):
            s_feature += custom(idx_f(sample))
        return np.asarray(s_feature)[self.feature_mask].tolist()

    def train(self, field_ids: List[int], data_path: Path) -> None:
        """Train the tillage classifier using the given field-IDs."""
        # Load in the training data first
        data = self.load_data(field_ids, data_path, balance=True)
        samples, labels = zip(*data)

        # Initialise the classifier
        self.clf = RandomForestClassifier()

        # Get the best feature map for the given data
        self.feature_mask = [True] * (4 * 2)  # 4 total features over 2 indices
        features = [self._get_features(sample) for sample in samples]
        sfs = SelectFromModel(
            estimator=self.clf,
        )
        sfs.fit(features, labels)
        self.feature_mask = sfs.get_support().tolist()

        # Train the classifier and save
        features = [self._get_features(sample) for sample in samples]  # Updated features
        self.clf.fit(features, labels)

    def eval(self, field_ids: List[int], data_path: Path) -> Tuple[List[bool], List[bool]]:
        """Evaluate the model on the given field-IDs."""
        # Load in the training data first
        data = self.load_data(field_ids, data_path, balance=False)
        samples, labels = zip(*data)

        # Make predictions on the given samples
        preds = []
        for sample in tqdm(samples, desc="Evaluating"):
            feature = self._get_features(sample)
            preds.append(self.clf.predict([feature])[0])  # type: ignore

        # Show stats and return (true-labels, predictions)
        print(f"Accuracy: {accuracy_score(labels, preds)}")
        print(f"  Recall: {recall_score(labels, preds)}")
        print(f"F1-score: {f1_score(labels, preds)}")
        return labels, preds

    def load_data(  # noqa C901
        self, field_ids: List[int], data_path: Path, balance: bool = False
    ) -> List[Tuple[Any, bool]]:
        """Load in the complete dataset (initial samples)."""
        data, labels = [], []

        # Fetch all field samples (and remove clouds)
        inputs = [(data_path / f"{i}", self.cloud_filter) for i in field_ids]
        with Pool(cpu_count() - 2) as p:
            samples = list(tqdm(p.imap(_load_sample, inputs), total=len(inputs), desc="Processing"))

        # Add all valid field samples to the dataset
        for s_data, s_label in samples:
            if s_label is None:
                continue
            s_data_comb: Dict[str, List[float]] = {}
            for sample in s_data.values():
                for band, values in sample.items():
                    if band not in s_data_comb:
                        s_data_comb[band] = []
                    s_data_comb[band] += values

            # Add to the dataset
            data.append(s_data_comb)
            labels.append(s_label)

        # Shuffle and balance the dataset
        if balance:
            labels_unique = sorted(set(labels))
            classes: Dict[str, List[Any]] = {label: [] for label in labels_unique}
            for d, l in zip(data, labels):
                classes[l].append((d, l))
            for label in labels_unique:
                shuffle(classes[label])
            data_comb = []  # Intertwine the classes
            for i in range(min(len(classes[label]) for label in labels_unique)):
                for label in labels_unique:
                    data_comb.append(classes[label][i])
        else:
            data_comb = list(zip(data, labels))
        return data_comb

    def load(self) -> None:
        """Load in a previously trained classifier with corresponding meta data, if exists."""
        if (self.models_path / "cloud_filter.pickle").is_file():
            self.cloud_filter = CloudFilter(model_path=self.models_path)
        if (self.models_path / "field_rf.pickle").is_file():
            with open(self.models_path / "field_rf.pickle", "rb") as f:
                self.clf = pickle.load(f)  # noqa S301
        if (self.models_path / "field_rf_meta.json").is_file():
            with open(self.models_path / "field_rf_meta.json", "r") as f:  # type: ignore
                self.feature_mask = json.load(f)["feature_mask"]

    def save(self) -> None:
        """Store the classifier and corresponding meta data."""
        assert self.clf is not None
        assert self.feature_mask is not None
        with open(self.models_path / "field_rf.pickle", "wb") as f:
            pickle.dump(self.clf, f)
        with open(self.models_path / "field_rf_meta.json", "w") as f:  # type: ignore
            json.dump({"feature_mask": self.feature_mask}, f, indent=2)  # type: ignore


def _load_sample(inp: Any) -> Tuple[Any, Optional[bool]]:  # noqa C901
    """Standalone function to load in a sample (multiprocessing purposes)."""
    read_path, cloud_filter = inp
    data: Dict[str, Any] = {}

    def add_data(new_data: Dict[str, Dict[str, List[float]]]) -> None:
        for date, sample in new_data.items():
            if cloud_filter is not None and cloud_filter(sample):
                continue
            if date in data:
                for band, values in sample.items():
                    data[date][band] += values
            else:
                data[date] = sample

    # Load in all the data
    with open(read_path / "samples/landsat7.json", "r") as f:
        add_data(json.load(f))
    with open(read_path / "samples/landsat8.json", "r") as f:
        add_data(json.load(f))
    with open(read_path / "samples/sentinel2.json", "r") as f:
        add_data(json.load(f))

    # Check if enough data samples remain, return empty result if not
    if len(data) < 5:
        return {}, None

    # Load in meta-data
    with open(read_path / "meta.json", "r") as f:
        tillage = json.load(f)["tillage"] != "No-Till"  # True if not tillage
    return data, tillage
