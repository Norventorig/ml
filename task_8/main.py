import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN


def define_outliers(param: pd.DataFrame, threshold: float = 1.5) -> int:
    """
    Функция рассчитывающая кол_во выбросов
    :param param:
    :param threshold:
    :return amount:
    """
    masks = {}

    for col in param.columns.to_list():
        q1 = np.quantile(param[col], 0.25)
        q3 = np.quantile(param[col], 0.75)

        iqr = q3 - q1

        lower = q1 - iqr * threshold
        upper = q3 + iqr * threshold

        masks[col] = (param[col] < lower) | (param[col] > upper)

    for col, mask in masks.items():
        mask = mask[mask.index.isin(param.index)]
        param = param[~mask]

    return len(param)


dataset = pd.read_csv('dataset.csv')


types = np.unique(dataset['Type'].values)
rec_nums = {f'Type {i}': len(dataset[dataset['Type'] == i]) for i in types}

print(f'{rec_nums}\n'
      'Наиболее распространен Type 2. Наименее распространен Type 6\n'
      'Type 2 > Type 1 > Type 7 > Type 3 > Type 5 > Type 6')


X = dataset.drop(axis=1, columns='Type')
Y = dataset['Type']

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=1)


model = RandomForestClassifier()
model.fit(X=train_x, y=train_y)


print(f'\nТочность модели: {model.score(X=test_x, y=test_y)}')


for col in dataset.drop('Type', axis=1).columns.to_list():
    plt.figure(figsize=(8, 6))
    plt.boxplot(dataset[col], vert=False)
    plt.title(f'Boxplot для признака "{col}"')
    plt.xlabel('Значения')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
# Единственный признак с нормальным распределением "Mg"

model_IF = IsolationForest(contamination=0.1)
model_DBS = DBSCAN(eps=1, min_samples=5)

df = dataset.copy()

df['Outlier_IF'] = model_IF.fit_predict(X)
df['Outlier_DBS'] = model_DBS.fit_predict(X)

print(f"\nloss data percentage according to Isolation Forest: {len(df[df['Outlier_IF'] == -1]) / len(df) * 100}")
print(f"loss data percentage according to DBSCAN: {len(df[df['Outlier_DBS'] == -1]) / len(df) * 100}")
print(f"loss data percentage according to IQR: {define_outliers(param=X) / len(df) * 100}")

df = df[df['Outlier_IF'] != -1]


X = df.drop(axis=1, columns=['Type', 'Outlier_IF', 'Outlier_DBS'])
Y = df['Type']

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=1)

model = RandomForestClassifier()
model.fit(X=train_x, y=train_y)

print(f'\nТочность модели: {model.score(X=test_x, y=test_y)}')
