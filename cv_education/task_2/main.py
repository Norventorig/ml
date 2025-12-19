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


def load_images(path, img_size, label_name='cat'):
    x, y = [], []
    directory = Path(path)

    for img_path in directory.iterdir():
        label = int(label_name in img_path.name)

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)

        img = vgg16.preprocess_input(img)

        x.append(img)
        y.append(label)

        aug_img = TRAIN_DATAGEN(image=img)['image']
        x.append(aug_img)
        y.append(label)

    return np.array(x), np.array(y)

