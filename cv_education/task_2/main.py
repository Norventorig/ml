import cv2
from pathlib import Path
import albumentations as A
from matplotlib import pyplot as plt

TRAIN_DATAGEN = A.Compose([A.HorizontalFlip(p=0.25),
                           A.VerticalFlip(p=0.4),
                           A.ShiftScaleRotate(shift_limit=0.05,
                                              scale_limit=0.1,
                                              rotate_limit=30,
                                              p=0.5),
                           A.ColorJitter(brightness=0.2,
                                         contrast=0.2,
                                         saturation=0.2,
                                         hue=0.1,
                                         p=0.3)])


def load_train_generator(path: str, img_size: tuple, augmentator):
    directory = Path(path)

    for i_path in directory.iterdir():
        img = cv2.imread(i_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        new_img = augmentator(image=img)

        yield new_img['image']


load_train_gen = load_train_generator(
    path=r"C:\Users\-\Documents\Downloads\dogs-vs-cats-redux-kernels-edition\train\train",
    img_size=(256, 256),
    augmentator=TRAIN_DATAGEN)
