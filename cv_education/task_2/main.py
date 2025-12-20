import cv2
from pathlib import Path
import albumentations as A
import numpy as np
import random

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


def load_train_images(path, img_size: tuple, labels: tuple):
    x, y = [], []
    directory = Path(path)

    for subdir in directory.iterdir():
        label = labels.index(subdir.name)

        for img_path in subdir.iterdir():
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)

            aug_img = TRAIN_DATAGEN(image=img)['image']

            img = vgg16.preprocess_input(img)
            aug_img = vgg16.preprocess_input(aug_img)

            x.append(img)
            y.append(label)

            x.append(aug_img)
            y.append(label)

    combined = list(zip(x, y))
    random.shuffle(combined)
    x, y = zip(*combined)

    return np.array(x), np.array(y)


x_train, y_train = load_train_images(
    path=r"C:\Users\123\Downloads\datasets\train",
    img_size=(224, 224),
    labels=("cat", "dog"))


base_model = vgg16.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

for layer in base_model.layers:
    layer.trainable = False

outputs = base_model.get_layer('block5_conv3').output
outputs = GlobalAveragePooling2D()(outputs)
outputs = Dense(1024, activation='relu')(outputs)
outputs = Dropout(0.5)(outputs)
outputs = Dense(1, activation='sigmoid')(outputs)

model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=32,
    callbacks=[
        EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
        ModelCheckpoint('model.keras', save_best_only=True)
    ]
)

