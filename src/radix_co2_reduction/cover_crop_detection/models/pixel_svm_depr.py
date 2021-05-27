"""Support Vector Machine classifier that detects cover crop on pixel-level."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sklearn.svm import SVC


class PixelSVM:
    """Support Vector Machine classifier that detects cover crop on pixel-level."""

    def __init__(
        self,
        models_path: Path,
    ) -> None:
        """
        Initialise the classifier.

        :param models_path: Path where models (classification, cloud filter) are stored
        """
        self.models_path = models_path
        self.clf: Optional[SVC] = None

    def __call__(self, field_sample: Dict[str, Dict[str, List[Optional[float]]]]) -> Any:
        """Predict if sample is tilled."""
        assert self.clf is not None
        raise NotImplementedError  # TODO

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
        raise NotImplementedError  # TODO
        s_feature = []
        for idx_f in (ndvi, ndti):
            s_feature += custom(idx_f(sample))
        return np.asarray(s_feature)[self.feature_mask].tolist()

    def train(self, field_ids: List[int], data_path: Path) -> None:
        """Train the tillage classifier using the given field-IDs."""
        raise NotImplementedError  # TODO
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
        raise NotImplementedError  # TODO
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
        raise NotImplementedError  # TODO
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
        raise NotImplementedError  # TODO
        if (self.models_path / "field_rf.pickle").is_file():
            with open(self.models_path / "field_rf.pickle", "rb") as f:
                self.clf = pickle.load(f)  # noqa S301
        if (self.models_path / "field_rf_meta.json").is_file():
            with open(self.models_path / "field_rf_meta.json", "r") as f:  # type: ignore
                self.feature_mask = json.load(f)["feature_mask"]

    def save(self) -> None:
        """Store the classifier and corresponding meta data."""
        assert self.clf is not None
        raise NotImplementedError  # TODO
        with open(self.models_path / "field_rf.pickle", "wb") as f:
            pickle.dump(self.clf, f)
        with open(self.models_path / "field_rf_meta.json", "w") as f:  # type: ignore
            json.dump({"feature_mask": self.feature_mask}, f, indent=2)  # type: ignore
