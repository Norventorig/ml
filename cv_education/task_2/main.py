import cv2
from pathlib import Path
import albumentations as A
import numpy as np

from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications.vgg16 import preprocess_input
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


class TrainGenerator(Sequence):
    def __init__(self, paths: tuple, batch_size: int, img_size: tuple, labels: tuple, aug_percentage: float = 0.0):
        self.paths = paths
        self.batch_size = batch_size
        self.img_size = img_size
        self.indexes = np.arange(len(paths))
        self.labels = labels
        self.aug_percentage = aug_percentage

        self.on_epoch_end()

    def __len__(self):
        return len(self.paths) // self.batch_size

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __getitem__(self, item):
        batch_x = np.zeros((self.batch_size, *self.img_size, 3), dtype=np.float32)
        batch_y = np.zeros(self.batch_size, dtype=np.int32)

        start = item * self.batch_size
        stop = start + self.batch_size
        for batch_index, i_index in enumerate(self.indexes[start:stop]):
            i_path = self.paths[i_index]
            i_label = self.labels[i_index]

            img = cv2.imread(str(i_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)

            if np.random.random() < self.aug_percentage:
                img = TRAIN_DATAGEN(image=img)['image']

            img = preprocess_input(img)

            batch_x[batch_index] = img
            batch_y[batch_index] = i_label

        return batch_x, batch_y


class ValidationGenerator(Sequence):
    def __init__(self, paths: tuple, batch_size: int, img_size: tuple, labels: tuple):
        self.paths = paths
        self.batch_size = batch_size
        self.img_size = img_size
        self.labels = labels

    def __len__(self):
        return len(self.paths) // self.batch_size

    def __getitem__(self, item):
        batch_x = np.zeros((self.batch_size, *self.img_size, 3), dtype=np.float32)
        batch_y = np.zeros(self.batch_size, dtype=np.int32)

        start = item * self.batch_size
        stop = start + self.batch_size
        for batch_index, i_index in enumerate(range(start, stop)):
            i_path = self.paths[i_index]
            i_label = self.labels[i_index]

            img = cv2.imread(str(i_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)

            img = preprocess_input(img)

            batch_x[batch_index] = img
            batch_y[batch_index] = i_label

        return batch_x, batch_y


train_dataset_path = Path(r"C:\Users\123\Downloads\datasets\train")

cat_train_dataset_path = train_dataset_path / 'cat'
dog_train_dataset_path = train_dataset_path / 'dog'

cat_paths = sorted(cat_train_dataset_path.iterdir(), key=lambda x: int(x.stem[4:]))
dog_paths = sorted(dog_train_dataset_path.iterdir(), key=lambda x: int(x.stem[4:]))

cat_labels = [0 for _ in cat_paths]
dog_labels = [1 for _ in dog_paths]

validation_split = 0.2

all_paths = cat_paths + dog_paths
all_labels = cat_labels + dog_labels

combined = list(zip(all_paths, all_labels))
np.random.shuffle(combined)
all_paths, all_labels = zip(*combined)

all_paths = tuple(all_paths)
all_labels = tuple(all_labels)

train_paths = all_paths[int(validation_split * len(all_paths)):]
train_labels = all_labels[int(validation_split * len(all_labels)):]

validation_paths = all_paths[:int(validation_split * len(all_paths))]
validation_labels = all_labels[:int(validation_split * len(all_labels))]

batch_size = 32
img_size = (224, 224)

train = TrainGenerator(paths=train_paths, batch_size=batch_size, img_size=img_size,
                       labels=train_labels)
validation = ValidationGenerator(paths=validation_paths, batch_size=batch_size, img_size=img_size,
                                 labels=validation_labels)

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

print('Начало обучения')
model.fit(
    train,
    validation_data=validation,
    epochs=30,
    batch_size=32,
    callbacks=[
        EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
        ModelCheckpoint('model.keras', save_best_only=True)
    ]
)

