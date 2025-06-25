import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


def remove_outliers(param: pd.DataFrame):
    for col in param.columns.to_list():
        q1 = np.quantile(param[col], 0.25)
        q3 = np.quantile(param[col], 0.75)

        iqr = q3 - q1

        lower = q1 - iqr * 1.5
        upper = q3 + iqr * 1.5

        param = param[(param[col] >= lower) & (param[col] <= upper)]

    return param


dataset = pd.read_csv('dataset.csv')

print(f'Кол-во пропусков в каждом признаке: \n{dataset.isnull().sum()}')
for col in dataset.columns.to_list():
    plt.figure(figsize=(8, 6))
    plt.boxplot(dataset[col], vert=False)
    plt.title(f'Boxplot для признака "{col}"')
    plt.xlabel('Значения')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

dataset = remove_outliers(dataset)

# Пропуски не были обнаружены
# Коробки с усами показали множественные выбросы к которым чувствительна нормализация
# Оставшиеся пригодными для нормализации признаки (в основном широта и возраст дома) в ней не нуждаются


X = dataset[['MedInc', 'AveOccup']]
Y = dataset['MedHouseVal']

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=1)


model = LinearRegression()
model.fit(X=train_x, y=train_y)
LR_score = model.score(X=test_x, y=test_y)

model = DecisionTreeRegressor()
model.fit(X=train_x, y=train_y)
DTR_score = model.score(X=test_x, y=test_y)
