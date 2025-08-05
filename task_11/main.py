import pandas as pd
from sklearn.model_selection import train_test_split


dataset = pd.read_csv('dataset.csv')
dataset = pd.get_dummies(data=dataset)


X = dataset.drop('HeartDisease', axis=1)
y = dataset['HeartDisease']

train_x, train_y, test_x, test_y = train_test_split(X, y, test_size=0.2, random_state=1)
