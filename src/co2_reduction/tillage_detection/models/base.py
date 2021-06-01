"""Base model for the tillage detection task."""
import pickle  # noqa S403
from pathlib import Path
from typing import Any, Dict, List, Optional


class BaseModel:
    """Base model for the tillage detection task."""

    def __init__(
        self,
        model_folder: Path,
    ) -> None:
        """Initialise the classifier."""
        self.model_folder = model_folder

    def __call__(
        self,
        sample: Dict[str, Dict[str, List[Optional[float]]]],
        year: Optional[int] = None,
    ) -> Any:
        """Predict if sample is tilled."""
        raise NotImplementedError

    def get_features(
        self,
        sample: Dict[str, Dict[str, List[Optional[float]]]],
        year: Optional[int] = None,
    ) -> Any:
        """Get the requested features from the given sample."""
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

    def load(self) -> None:
        """Load in a previously trained classifier with corresponding meta data, if exists."""
        raise NotImplementedError

    def save(self) -> None:
        """Store the classifier and corresponding meta data."""
        raise NotImplementedError
