"""1-dimensional CNN that analyses a single pixel over time."""
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.radix_co2_reduction.data import (
    BANDS,
    datetime_to_int,
    get_label,
    load_pixel_data,
    process_sample_pixel,
)


class CnnDataset(Dataset):  # type: ignore
    """Dataset used to feed data to the classifier during training."""

    def __init__(
        self,
        field_ids: List[int],
        data_path: Path,
        balance: bool = False,
    ) -> None:
        """Initialise the dataset."""
        # Load in the data
        self.data, self.labels = load_data(field_ids=field_ids, data_path=data_path)

        # Balance, if requested
        if balance:
            sm = RandomOverSampler(random_state=42)
            data_idx = [[idx] for idx in range(len(self.data))]
            data_idx, labels = sm.fit_resample(data_idx, self.labels)
            self.data = [self.data[i[0]] for i in data_idx]
            self.labels = np.asarray(labels)

    def __len__(self) -> int:
        """Total size of the dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Get the item under the requested index from the dataset."""
        data = torch.from_numpy(self.data[idx].astype(np.float32))  # type: ignore
        label = self.labels[idx]
        return data, label


class CNN(pl.LightningModule):
    """Convolutional Neural Network used to classify pixel-level spectral data."""

    N_CONV1: int = 32
    N_CONV2: int = 64
    R_DROPOUT: float = 0.1
    lr: float = 5e-4
    loss: Any = nn.CrossEntropyLoss()  # type: ignore

    def __init__(self) -> None:
        """Create the CNN classifier."""
        super().__init__()  # type: ignore
        self.conv1 = nn.modules.Conv1d(len(BANDS), self.N_CONV1, kernel_size=5, padding=2)
        self.pool = nn.modules.MaxPool1d(2, padding=1)
        self.conv2 = nn.modules.Conv1d(self.N_CONV1, self.N_CONV2, kernel_size=3, padding=1)
        self.adapt_pool = nn.modules.AdaptiveMaxPool1d(1)
        self.dropout = nn.modules.Dropout(p=self.R_DROPOUT)
        self.fc = nn.modules.Linear(self.N_CONV2, 2)

    def forward(self, x: Any) -> Any:  # type: ignore
        """Single forward through the network."""
        x = self.pool(torch.relu(self.conv1(x)))  # type: ignore
        x = self.adapt_pool(torch.relu(self.conv2(x)))  # type: ignore
        x = x.view(-1, self.N_CONV2)
        x = self.dropout(x)
        x = torch.relu(self.fc(x))  # type: ignore
        return x

    def training_step(self, batch: Any, batch_idx: Any) -> Any:  # type: ignore
        """Train on the given batch."""
        # training_step defines the train loop
        x, y = batch
        y = torch.flatten(y)  # type: ignore
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)

        # Logging to TensorBoard by default
        yy = [bool(a) for a in y]
        yy_hat = [bool(torch.argmax(a)) for a in y_hat]  # type: ignore
        self.log("train_acc", accuracy_score(yy, yy_hat))
        return loss

    def validation_step(self, batch: Any, batch_idx: Any) -> Any:  # type: ignore
        """Validate the given batch."""
        # training_step defines the train loop
        x, y = batch
        y = torch.flatten(y)  # type: ignore
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss)

        # Logging to TensorBoard by default
        yy = [bool(a) for a in y]
        yy_hat = [bool(torch.argmax(a)) for a in y_hat]  # type: ignore
        return {
            "loss": loss,
            "data": (yy, yy_hat),
        }

    def validation_epoch_end(self, outputs: Any) -> None:
        """Use the test_step outputs to determine metrics as accuracy, recall, and f1 score."""
        if not outputs:
            return
        if not isinstance(outputs, list):
            outputs = [outputs]

        # Collect all data and calculate accuracy
        y, y_hat = [], []
        for output in outputs:
            a, b = output["data"]
            y += a
            y_hat += b
        assert len(y) == len(y_hat)
        self.log("val_acc", accuracy_score(y, y_hat))

    def test_step(self, batch: Any, batch_idx: Any) -> Any:  # type: ignore
        """Evaluate the given batch, extract other relevant information from the evaluation step."""
        # training_step defines the train loop
        x, y = batch
        y = torch.flatten(y)  # type: ignore
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log("test_loss", loss)

        # Logging to TensorBoard by default
        yy = [bool(a) for a in y]
        yy_hat = [bool(torch.argmax(a)) for a in y_hat]  # type: ignore
        accuracy = accuracy_score(yy, yy_hat)
        self.log("test_acc", accuracy)

    def configure_optimizers(self) -> Any:
        """Optimizers used during training."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class PixelCNN:
    """One dimensional CNN classification model to predict tillage events on pixel-level."""

    def __init__(
        self,
        models_path: Path,
    ):
        """
        Initialise the classifier.

        :param models_path: Path where models (classification, cloud filter) are stored
        """
        self.models_path = models_path
        self.clf: Optional[CNN] = None
        self.thr: Optional[float] = 0.5
        self._trainer: Optional[pl.Trainer] = None
        self.load()

    def __call__(
        self, field_sample: Dict[str, Dict[str, List[Optional[float]]]], year: int
    ) -> Optional[bool]:
        """Check if the field (sample) contains a tillage event or not."""
        probs = self.get_pixel_probs(field_sample=field_sample, year=year)
        if not probs:
            return None

        # Categorise the predicted probabilities (high probabilities carry larger weight)
        return (sum(probs) / len(probs)) >= self.thr  # type: ignore

    def get_pixel_probs(
        self, field_sample: Dict[str, Dict[str, List[Optional[float]]]], year: int
    ) -> List[float]:
        """Get the probabilities of each pixel that it contains a pixel-event over time."""
        # Ensure that the classifier is in evaluation mode
        if self.clf.training:  # type: ignore
            self.clf.eval()  # type: ignore

        data = process_sample_pixel(
            sample=field_sample,
            start_idx=datetime_to_int(f"{year - 1}-11-01"),
            end_idx=datetime_to_int(f"{year}-04-30"),
        )

        # Return None if no data found
        if not data:
            return []
        pred = self.clf(torch.from_numpy(np.asarray(data, dtype=np.float32)))  # type: ignore
        return [float(p[1]) for p in torch.softmax(pred, dim=1)]  # type: ignore  # p[1] == prob(True/Tillage)

    def create_trainer(self) -> None:
        """Create a trainer for the network."""
        callbacks = [
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,  # val_loss calculated 4x/epoch
            ),
            pl.callbacks.ModelCheckpoint(
                monitor="val_loss",
                dirpath=self.models_path,
                filename="pixel_cnn",
                save_top_k=1,
                mode="min",
            ),
        ]
        self._trainer = pl.Trainer(
            callbacks=callbacks,
            val_check_interval=0.5,  # Check validation every half training epoch
        )

    def train(
        self,
        field_ids: List[int],
        data_path: Path,
        batch_size: int = 4096,
        val_ratio: float = 0.1,
    ) -> None:
        """Train the classification model."""
        # Split into a training and validation set first
        field_labels = [get_label(data_path / f"{i}") for i in field_ids]
        train, val = train_test_split(
            field_ids,
            test_size=val_ratio,
            stratify=field_labels,
            random_state=42,
        )

        # Load in the data
        train_dataset = CnnDataset(field_ids=train, data_path=data_path, balance=True)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            # collate_fn=my_collate,
        )
        val_dataset = CnnDataset(field_ids=val, data_path=data_path, balance=True)
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=True,
            # collate_fn=my_collate,
        )

        # Remove previously saved model
        (self.models_path / "pixel_cnn.ckpt").unlink(missing_ok=True)

        # Create the classifier and setup a trainer
        self.clf = CNN()
        if self._trainer is None:
            self.create_trainer()

        # Train the model
        self._trainer.fit(  # type: ignore
            self.clf,
            train_dataloader=train_loader,
            val_dataloaders=val_loader,
        )

        # Load back in the best version of the model (early stopped)
        self.load()

    def eval(
        self,
        field_ids: List[int],
        data_path: Path,
        batch_size: int = 4096,
    ) -> None:
        """Evaluate the classification model."""
        assert self.clf is not None

        # Load in the data
        test_dataset = CnnDataset(field_ids=field_ids, data_path=data_path)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            # collate_fn=my_collate,
        )

        # Ensure the trainer exists
        if self._trainer is None:
            self.create_trainer()

        # Evaluate the network
        results = self._trainer.test(  # type: ignore
            self.clf,
            test_dataloaders=test_loader,
            verbose=False,
        )[0]
        print(f"    Loss: {results['test_loss']:.5f}")
        print(f"Accuracy: {results['test_acc']:.5f}")

    def load(self) -> None:
        """Load in a previously created model."""
        if (self.models_path / "pixel_cnn.ckpt").is_file():
            self.clf = CNN.load_from_checkpoint(str(self.models_path / "pixel_cnn.ckpt"))


def load_data(
    field_ids: List[int],
    data_path: Path,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Load in the data samples corresponding the field-IDs."""
    # Fetch all field samples
    inputs = [data_path / f"{i}" for i in field_ids]
    with Pool(cpu_count() - 2) as p:
        samples = list(tqdm(p.imap(load_pixel_data, inputs), total=len(inputs), desc="Processing"))

    # Unfold the data
    data_lst: List[np.ndarray] = []
    labels_lst: List[int] = []
    for s_data, s_label in samples:
        if s_label is None:
            continue
        for d in s_data:
            if d.shape[1] < 5:
                continue
            data_lst.append(d)
            labels_lst.append(int(s_label))

    # Convert to numpy arrays and return
    return data_lst, np.asarray(labels_lst).reshape((len(labels_lst), 1))
