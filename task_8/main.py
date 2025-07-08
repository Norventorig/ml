import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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
