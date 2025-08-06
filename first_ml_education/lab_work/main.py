from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt


dataset = fetch_ucirepo(id=40).data.features
dataset = pd.get_dummies(data=dataset, columns=['zone', 'landmass', 'language', 'mainhue', 'botright', 'topleft'])

X = dataset.drop('religion', axis=1)
y = dataset['religion']

