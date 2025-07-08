import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest

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

model = IsolationForest(contamination=0.1)
model.fit(dataset)

df = dataset.copy()
df['is_anomaly'] = model.predict(dataset)
df['is_anomaly'] = df['is_anomaly'].map({1: 0, -1: 1})

print(df)
