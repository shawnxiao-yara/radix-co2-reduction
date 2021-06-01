"""Random Forest classifier that detects tillage events on field-level."""
import json
import pickle  # noqa S403
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from imblearn.over_sampling import RandomOverSampler
from radix_co2_reduction.earth_engine.cloud_filter import CloudFilter
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report

from src.co2_reduction.data import BANDS
from src.co2_reduction.tillage_detection.features import custom
from src.co2_reduction.tillage_detection.index import (
    evi,
    ndti,
    ndvi,
    nir,
    sar_vh,
    sar_vv,
    swir1,
    swir2,
)
from src.co2_reduction.tillage_detection.models.base import BaseModel


class FieldRF(BaseModel):
    """Random Forest classifier that predicts on field-level."""

    FEATURE_BANDS = (nir, swir1, swir2, sar_vv, sar_vh, ndvi, evi, ndti)

    def __init__(
        self,
        model_folder: Path,
    ) -> None:
        """
        Initialise the classifier.

        :param folder: Directory where models (classification, cloud filter) are stored
        """
        super().__init__(model_folder=model_folder)
        self.feature_mask: Optional[List[bool]] = None
        self.clf: Optional[RandomForestClassifier] = None
        self.cloud_filter: Optional[CloudFilter] = None
        self.load()

    def __call__(
        self,
        sample: Dict[str, Dict[str, List[Optional[float]]]],
        year: Optional[int] = None,
    ) -> Any:
        """Predict if sample is tilled."""
        assert self.clf is not None
        assert self.feature_mask is not None

        # Extract features and make a prediction
        feature = self.get_features(sample)
        return self.clf.predict([feature])[0]

    def get_features(
        self,
        sample: Dict[str, Dict[str, List[Optional[float]]]],
        year: Optional[int] = None,
    ) -> Any:
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
        for idx_f in self.FEATURE_BANDS:
            s_feature += custom(idx_f(collapsed))
        return np.asarray(s_feature)[self.feature_mask].tolist()

    def init_feature_mask(self) -> None:
        """Re-initialise the feature mask by setting all its values to True."""
        self.feature_mask = [True] * (4 * len(self.FEATURE_BANDS))  # 4 total features / band

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
        Train the tillage detection classifier.

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
    ) -> List[bool]:
        """
        Evaluate the tillage detection classifier.

        :param features: Input features created using get_features
        :param labels: List of tillage labels corresponding the input features
        """
        # Make predictions on the given samples
        preds = self.clf.predict(features)  # type: ignore

        # Show stats and return (true-labels, predictions)
        print("Evaluation result:")
        print(classification_report(labels, preds))
        return preds  # type: ignore

    def load(self) -> None:
        """Load in a previously trained classifier with corresponding meta data, if exists."""
        if (self.model_folder / "cloud_filter.pickle").is_file():
            self.cloud_filter = CloudFilter(model_path=self.model_folder)
        if (self.model_folder / "td_field_rf.pickle").is_file():
            with open(self.model_folder / "td_field_rf.pickle", "rb") as f:
                self.clf = pickle.load(f)  # noqa S301
        if (self.model_folder / "td_field_rf_meta.json").is_file():
            with open(self.model_folder / "td_field_rf_meta.json", "r") as f:  # type: ignore
                self.feature_mask = json.load(f)["feature_mask"]

    def save(self) -> None:
        """Store the classifier and corresponding meta data."""
        assert self.clf is not None
        assert self.feature_mask is not None
        with open(self.model_folder / "td_field_rf.pickle", "wb") as f:
            pickle.dump(self.clf, f)
        with open(self.model_folder / "td_field_rf_meta.json", "w") as f:  # type: ignore
            json.dump({"feature_mask": self.feature_mask}, f, indent=2)  # type: ignore
