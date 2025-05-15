import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


scaler = StandardScaler()
dataset = pd.read_csv('dataset.csv')

dataset = dataset[['height', 'weight', 'sex']]
dataset = dataset.dropna()

dataset['sex'] = (dataset['sex'] == 'male').astype(int)

x = dataset[['height', 'weight']]
y = dataset['sex']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = LogisticRegression()
model.fit(x_train, y_train)
