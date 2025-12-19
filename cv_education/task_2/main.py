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


x_train, y_train = load_images(
    path=r"C:\Users\123\Downloads\train",
    img_size=(224, 224)
)


base_model = vgg16.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.get_layer('block5_conv3').output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid')(x)

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

