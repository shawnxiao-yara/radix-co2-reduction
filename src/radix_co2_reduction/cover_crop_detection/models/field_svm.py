"""Support Vector Machine that detects cover crop on field-level."""
import pickle  # noqa S403
from pathlib import Path
from typing import Any, Dict, List, Optional

from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.svm import SVC

from src.radix_co2_reduction.cover_crop_detection.features import ndvi_feature


class FieldSVM:
    """Support Vector Machine that detects cover crop on field-level."""

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
        self.load()

    def __call__(self, sample: Dict[str, Dict[str, List[Optional[float]]]], year: int) -> Any:
        """Predict if sample is tilled."""
        assert self.clf is not None

        # Extract features and make a prediction
        feature = ndvi_feature(sample, year=year)
        return self.clf.predict([feature])[0]

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
        self.clf = SVC(
            random_state=42,
        )
        self.clf.fit(features, labels)

    def eval(
        self,
        features: List[Any],
        labels: List[bool],
    ) -> List[bool]:
        """Evaluate the model on the given field-IDs."""
        # Make predictions on the given samples
        preds = self.clf.predict(features)  # type: ignore

        # Show stats and return (true-labels, predictions)
        print(f"Accuracy: {accuracy_score(labels, preds)}")
        print(f"  Recall: {recall_score(labels, preds)}")
        print(f"F1-score: {f1_score(labels, preds)}")
        return preds  # type: ignore

    def load(self) -> None:
        """Load in a previously trained classifier with corresponding meta data, if exists."""
        if (self.models_path / "cc_field_svm.pickle").is_file():
            with open(self.models_path / "cc_field_svm.pickle", "rb") as f:
                self.clf = pickle.load(f)  # noqa S301

    def save(self) -> None:
        """Store the classifier and corresponding meta data."""
        assert self.clf is not None
        with open(self.models_path / "cc_field_svm.pickle", "wb") as f:
            pickle.dump(self.clf, f)
