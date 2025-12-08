import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


train = np.loadtxt('train.csv', skiprows=1, delimiter=',')
test = np.loadtxt('test.csv', skiprows=1, delimiter=',')

labels = train[:, 0]
train_img = np.reshape(train[:, 1:], (train.shape[0], 28, 28))
test = np.reshape(test, (test.shape[0], 28, 28))


def pixels_gradients(array: np.ndarray):
    len_y = array.shape[0]
    len_x = array.shape[1]

    gx = np.zeros(array.shape)
    gy = np.zeros(array.shape)

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    for y_vect in range(0, len_y-2):
        for x_vect in range(0, len_x-2):
            matrix = array[y_vect:y_vect+3, x_vect:x_vect+3]
            gx[y_vect+1, x_vect+1] = sum([matrix[i_y, i_x] * sobel_x[i_y, i_x] for i_y in range(3) for i_x in range(3)])
            gy[y_vect+1, x_vect+1] = sum([matrix[i_y, i_x] * sobel_y[i_y, i_x] for i_y in range(3) for i_x in range(3)])

    return gy, gx


def calculate_magnitude(array_1: np.ndarray, array_2: np.ndarray):
    return np.sqrt(array_1**2 + array_2**2)


def calculate_orientation(array_1: np.ndarray, array_2: np.ndarray):
    return np.degrees(np.arctan2(array_1, array_2)) % 180


def cell_histograms(orientation: np.ndarray, magnitude: np.ndarray, nbins: int, cell_size: int):
    histograms = np.zeros((orientation.shape[0] // cell_size, orientation.shape[1] // cell_size, nbins))
    bin_size = 180 // nbins

    for y_cord_cell, i_y in enumerate(range(0, magnitude.shape[0], cell_size)):
        for x_cord_cell, i_x in enumerate(range(0, magnitude.shape[1], cell_size)):
            cell_magnitude = magnitude[i_y:i_y+cell_size, i_x: i_x+cell_size]
            cell_orientation = orientation[i_y:i_y+cell_size, i_x: i_x+cell_size]

            for y_pixel in range(0, cell_size):
                for x_pixel in range(0, cell_size):
                    angle = cell_orientation[y_pixel, x_pixel]

                    index = int(angle // bin_size)
                    histograms[y_cord_cell, x_cord_cell, index] += cell_magnitude[y_pixel, x_pixel]

    return histograms


def normalize_histograms(histograms: np.ndarray, block_size: int):
    normalized = list()

    for i_y in range(histograms.shape[0] - block_size):
        for i_x in range(histograms.shape[1] - block_size):
            block = histograms[i_y: i_y+block_size, i_x: i_x+block_size].flatten()

            norm = block / (np.linalg.norm(block) + 1e-6)
            normalized.append(norm)

    return np.concatenate(normalized)


def prepare_img(array: np.ndarray, nbins: int = 9, cell_size: int = 7, block_size: int = 2):
    gy, gx = pixels_gradients(array=array)
    magnitude = calculate_magnitude(array_1=gy, array_2=gx)
    orientation = calculate_orientation(array_1=gy, array_2=gx)
    histograms = cell_histograms(orientation=orientation, magnitude=magnitude, nbins=nbins, cell_size=cell_size)
    return normalize_histograms(histograms=histograms, block_size=block_size)


y = labels
x = [prepare_img(array=i) for i in train_img]
test = [prepare_img(array=i) for i in test]

# params = {'n_estimators': [100, 300, 500],
#           'max_depth': [10, 20, 30],
#           'min_samples_split': [2, 5, 10],
#           'min_samples_leaf': [1, 2, 4],
#           'max_features': ['auto', 'sqrt', 'log2'],
#           'bootstrap': [True, False]}
#
# grid = GridSearchCV(RandomForestClassifier(), params)
# grid.fit(x_train, y_train)

best_params = {'bootstrap': False,
               'max_depth': 10,
               'max_features': 'log2',
               'min_samples_leaf': 1,
               'min_samples_split': 2,
               'n_estimators': 300}
model = RandomForestClassifier(**best_params)
model.fit(x, y)

y_pred = model.predict(test)

data = {'ImageId': [i for i in range(1, len(y_pred)+1)], 'Label': [int(i) for i in y_pred]}
pd.DataFrame(data=y_pred).to_csv('results.csv', index=False)

# Итоговая метрика: 0.928
