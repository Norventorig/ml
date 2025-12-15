import cv2
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TRAIN_DATAGEN = ImageDataGenerator(rotation_range=180,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   shear_range=0.1,
                                   channel_shift_range=10.0)


def load_train_generator(path, img_size, augmentator):
    directory = Path(path)
    for i_path in directory.iterdir():
        img = cv2.imread(i_path, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, img_size)
        augmentator.flow(img.reshape((1, *img_size, 3)), batch_size=1)

        yield img


load_train_gen = load_train_generator(
    path=r"C:\Users\-\Documents\Downloads\dogs-vs-cats-redux-kernels-edition\train\train",
    img_size=(256, 256),
    augmentator=TRAIN_DATAGEN)
