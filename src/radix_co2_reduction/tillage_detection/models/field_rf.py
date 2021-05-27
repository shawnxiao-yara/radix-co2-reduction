"""Random Forest classifier that detects tillage events on field-level."""
import json
import pickle  # noqa S403
from multiprocessing import Pool, cpu_count
from pathlib import Path
from random import shuffle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, f1_score, recall_score
from tqdm import tqdm

from radix_co2_reduction.earth_engine.cloud_filter import CloudFilter
from src.radix_co2_reduction.data import BANDS, load_field_data
from src.radix_co2_reduction.tillage_detection.features import custom
from src.radix_co2_reduction.tillage_detection.index import evi, ndti, ndvi, nir, swir1, swir2


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

    def __call__(self, sample: Dict[str, Dict[str, List[Optional[float]]]]) -> Any:
        """Predict if sample is tilled."""
        assert self.clf is not None
        assert self.feature_mask is not None

        # Extract features and make a prediction
        feature = self.get_features(sample)
        return self.clf.predict([feature])[0]

    def get_features(self, sample: Dict[str, Dict[str, List[Optional[float]]]]) -> Any:
        """Get the requested features from the given sample."""
        assert self.feature_mask is not None

        # Collapse the sample
        collapsed: Dict[str, List[Optional[float]]] = {band: [] for band in BANDS}
        for value in sample.values():
            # TODO: Cloud filter is too slow to use!
            # if self.cloud_filter is not None and self.cloud_filter(value):
            #     continue
            for band in BANDS:
                collapsed[band] += value[band]

        # Create the features
        s_feature = []
        for idx_f in (nir, swir1, swir2, ndvi, evi, ndti):
            s_feature += custom(idx_f(collapsed))
        return np.asarray(s_feature)[self.feature_mask].tolist()

    def init_feature_mask(self) -> None:
        """Re-initialise the feature mask by setting all its values to True."""
        self.feature_mask = [True] * (4 * 6)  # 4 total features over 6 indices/bands

    def optimise_feature_mask(
        self,
        features: List[Any],
        labels: List[bool],
    ) -> None:
        """
        Optimise the feature mask by utilising the features with most variety between classes.

        Note: init_feature_mask must be run before creating the input features.

        :param features: Input features that are the result of a recently initialised feature mask
        :param labels: List of tillage labels corresponding the input features
        """
        # Temporal classifier
        _clf = RandomForestClassifier()
        sfs = SelectFromModel(
            estimator=_clf,
        )
        sfs.fit(features, labels)
        self.feature_mask = sfs.get_support().tolist()

    def train(
        self,
        features: List[Any],
        labels: List[bool],
    ) -> None:
        """
        Train the tillage classifier using the given field-IDs.

        :param features: Input features created using get_features
        :param labels: List of tillage labels corresponding the input features
        """
        # Balance the features by oversampling the underrepresented class
        sm = RandomOverSampler(random_state=42)
        features, labels = sm.fit_resample(features, labels)

        # Train the classifier
        self.clf = RandomForestClassifier()
        self.clf.fit(features, labels)

    def eval(
        self,
        features: List[Any],
        labels: List[bool],
    ) -> Tuple[List[bool], List[bool]]:
        """Evaluate the model on the given field-IDs."""
        # Make predictions on the given samples
        preds = []
        for feature in tqdm(features, desc="Evaluating"):
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
            samples = list(
                tqdm(p.imap(load_field_data, inputs), total=len(inputs), desc="Processing")
            )

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

        # Shuffle and balance the dataset  TODO: Use sklearn +balance on features
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
