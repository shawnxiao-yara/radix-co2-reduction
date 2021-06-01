"""Classifier class to filter out clouds."""
import pickle  # noqa: S403
from pathlib import Path
from random import sample as sample_list
from random import shuffle
from typing import Any, Dict, List, Optional, Tuple

from sklearn.metrics import f1_score
from sklearn.svm import SVC

BANDS = ("B", "G", "R", "NIR", "SWIR1", "SWIR2")


class CloudFilter:
    """Classifier class to filter out clouds."""

    def __init__(self, model_path: Path):
        self.clf: Optional[SVC] = None
        self.thr: float = 0.72
        self.path = model_path
        self.load()

    def __call__(self, sample: Dict[str, List[float]]) -> Any:
        """
        Test if image contains cloud.

        :param sample: Normalised sample that has a list of values for each RGB band (0..1)
        :return: Whether or not clouds are recognised
        """
        assert self.clf is not None
        features = _extract_features(sample)
        if len(features) == 0:  # If no data, assume (masked out) cloud
            return True
        predictions = self.clf.predict(features)
        return sum(predictions) >= self.thr * len(features)

    def train(self, cloud_data: Dict[str, Any]) -> None:
        """Train a SVM classifier on the given cloud data."""
        # Format data first
        data = format_data(cloud_data, balance=True)

        # Train the pixel-classifier
        x, y = zip(*data)
        self.clf = SVC(random_state=42)
        self.clf.fit(x, y)

    def calc_best_thr(self, cloud_data: Dict[str, Any]) -> float:
        """Calculate the best possible threshold value."""
        eval_samples, true = zip(
            *[(s, True) for s in cloud_data["cloudy"]] + [(s, False) for s in cloud_data["clear"]]
        )
        pixel_preds = []
        for sample in eval_samples:
            features = _extract_features(sample)
            preds = self.clf.predict(features)  # type: ignore
            pixel_preds.append(sum(preds) / len(features))

        # Get best F1-score
        best_thr, best_score = 0.0, 0.0
        for thr_step in range(100):
            thr = thr_step / 100
            preds = [p >= thr for p in pixel_preds]
            score = f1_score(true, preds)
            if score > best_score:
                best_thr, best_score = thr, score
        return best_thr

    def exists(self) -> bool:
        """Check if model exists."""
        return self.clf is not None

    def save(self) -> None:
        """Save the model."""
        with open(self.path / "cloud_filter.pickle", "wb") as f:
            pickle.dump(self.clf, f)
            print("Saved cloud-filter!")

    def load(self) -> None:
        """Load the model, if exists."""
        if (self.path / "cloud_filter.pickle").is_file():
            with open(self.path / "cloud_filter.pickle", "rb") as f:
                self.clf = pickle.load(f)  # noqa: S301
            print("Loaded existing cloud-filter!")


def format_data(cloud_data: Dict[str, Any], balance: bool = True) -> List[Tuple[List[float], bool]]:
    """Create the training data."""
    # Extract all features first
    features_cloud = []
    for sample in cloud_data["cloudy"]:
        s_features = _extract_features(sample)
        features_cloud += s_features
    features_clear = []
    for sample in cloud_data["clear"]:
        s_features = _extract_features(sample)
        features_clear += s_features

    # Upsample cloudy until roughly the same datasets
    while balance and len(features_cloud) < 0.9 * len(features_clear):
        features_cloud += sample_list(features_cloud, 200)

    # Combine, shuffle and return
    features = features_cloud + features_clear
    labels = [True] * len(features_cloud) + [False] * len(features_clear)
    data = list(zip(features, labels))
    shuffle(data)
    return data


def _extract_features(sample: Dict[str, List[float]]) -> List[List[float]]:
    """Extract all pixel-features from the sample."""
    features = []
    for p_idx in range(len(sample["R"])):
        if any(sample[b][p_idx] is None for b in BANDS):
            continue
        features.append([sample[b][p_idx] for b in BANDS])
    return features
