"""Dataset class."""
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.utils.data
from PIL import Image

from src.radix_co2_reduction.field_detector.mask_rcnn.transforms import get_transform


class Dataset(torch.utils.data.Dataset):  # type: ignore
    """Dataset used to train the Mask RCNN model."""

    def __init__(
        self,
        path: Path,
    ) -> None:
        """Initialise the dataset."""
        self.path = path
        self.transforms = get_transform()
        self.field_paths = glob(str(self.path / "fields/*.png"))
        self.mask_paths = glob(str(self.path / "masks/*.png"))

    def __getitem__(self, idx: int) -> Any:
        """Get the item at the given index from the dataset."""
        # load images and masks
        img = Image.open(self.field_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx])

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)  # type: ignore
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)  # type: ignore
        masks = torch.as_tensor(masks, dtype=torch.uint8)  # type: ignore

        image_id = torch.tensor([idx])  # type: ignore
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])  # type: ignore
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)  # type: ignore

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }
        img, target = self.transforms(img, target)
        return img, target

    def __len__(self) -> int:
        """Get the size of the dataset."""
        return len(self.field_paths)
