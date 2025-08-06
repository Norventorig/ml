import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree


def remove_outliers(param: pd.DataFrame):
    for column in param.columns.to_list():
        q1 = np.quantile(param[column], 0.25)
        q3 = np.quantile(param[column], 0.75)

        iqr = q3 - q1

        lower = q1 - iqr * 1.5
        upper = q3 + iqr * 1.5

        param = param[(param[column] >= lower) & (param[column] <= upper)]

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


plt.figure(figsize=(12, 8))
plot_tree(
    model,
    feature_names=X.columns.to_list(),
    precision=2,
    filled=True,
    rounded=True,
    fontsize=10,
    max_depth=3
)
plt.title("Первые 3 уровня дерева решений")
plt.show()


best_DTR_score = \
    max(
        [
            ((model := DecisionTreeRegressor(max_depth=i_depth).fit(X=train_x, y=train_y)).score(X=test_x, y=test_y),
             i_depth)
            for i_depth in range(15, 2, -1)
        ]
    )[0]


print(f'\n\nЛинейная регрессия: {LR_score}'
      f'\nДрево решений: {DTR_score}'
      f'\nМодифицированное древо решений: {best_DTR_score}')
