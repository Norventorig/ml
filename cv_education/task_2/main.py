import cv2
from pathlib import Path
import albumentations as A
import numpy as np

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.applications import vgg16


TRAIN_DATAGEN = A.Compose([
    A.HorizontalFlip(p=0.25),
    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.1,
        rotate_limit=30,
        p=0.5
    ),
    A.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1,
        p=0.3
    )
])

