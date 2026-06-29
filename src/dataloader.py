"""
Author: Duy-Phuong Dao
Email: phuongdd.1997@gmail.com (or duyphuongcri@gmail.com)
"""

import torch
import monai
from torch.utils.data import DataLoader
import numpy as np

from monai.transforms import (
    Compose,
    LoadImaged,
    # AddChanneld,
    # SpatialPadd,
    ToTensord,
    # RandRotated,
    # RandZoomd,
    # RandSpatialCropd,
    # ConcatItemsd,
    # MapLabelValued,
    # MapTransform
)

def get_transforms_text():
    train_target_transforms = Compose(
        [
            LoadImaged(keys=["text"], reader='NumpyReader', image_only=False),
            ToTensord(keys=["text"]),
        ]
    )

    return train_target_transforms


def cache_transformed_text(train_files):
    train_transforms = get_transforms_text()
    train_ds = monai.data.CacheDataset(
        data=train_files, transform=train_transforms, cache_rate=0.0
    )
    return train_ds
