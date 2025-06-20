import pandas
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(x: pandas.DataFrame, y: pandas.Series, size: float = 0.2, rs: int = None):
    """
    Функция, которая рассчитывает и возвращает метрики RMSE и R2
    :param x:
    :param y:
    :param size:
    :param rs:
    :return:
    """
    in_train_x, in_test_x, in_train_y, in_test_y = train_test_split(x, y, test_size=size, random_state=rs)
    model.fit(X=in_train_x, y=in_train_y)
    in_predictions = model.predict(X=in_test_x)

    in_rmse = np.sqrt(np.mean((in_predictions - in_test_y) ** 2))
    in_r2 = r2_score(y_true=in_test_y, y_pred=in_predictions)

    return in_rmse, in_r2


def make_boxplot(param: pandas.Series, title='Коробка с усами'):
    """
    Функция выводит график коробка с усами
    :param param:
    :param title:
    :return:
    """
    lower, upper = define_outliers(param=param)

    plt.figure(figsize=(10, 6))
    plt.boxplot(param, vert=False, patch_artist=True)

    plt.axvline(lower, color='r', linestyle='--', alpha=0.7, label=f'Lower bound: {lower:.2f}')
    plt.axvline(upper, color='g', linestyle='--', alpha=0.7, label=f'Upper bound: {upper:.2f}')

    plt.title(title, fontsize=14)
    plt.grid(axis='x', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def define_outliers(param: pandas.Series, threshold: float = 1.5):
    """
    Функция рассчитывающая нижнюю и верхнюю границы выбросовAdd commentMore actions
    :param param:
    :param threshold:
    :return lower, upper:
    """
    q1 = np.quantile(param, 0.25)
    q3 = np.quantile(param, 0.75)

    iqr = q3 - q1

    lower = q1 - iqr * threshold
    upper = q3 + iqr * threshold

    return lower, upper


# :Attribute Information:
#     - MedInc        median income in block group
#     - HouseAge      median house age in block group
#     - AveRooms      average number of rooms per household
#     - AveBedrms     average number of bedrooms per household
#     - Population    block group population
#     - AveOccup      average number of household members
#     - Latitude      block group latitude
#     - Longitude     block group longitude


data = fetch_california_housing(as_frame=True)
df = pd.DataFrame(data=data.data, columns=data.feature_names)
df['MedHouseVal'] = data.target


null_count = sum(df.isnull().sum().to_list())
print('Пропуски есть' if 0 != null_count else 'Пропусков нет')
df.dropna(axis=1, inplace=True)

X = df[['HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude', 'MedInc']]
Y = df['MedHouseVal']

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=1)

scaler = StandardScaler()
train_x = scaler.fit_transform(X=train_x)
test_x = scaler.transform(X=test_x)

model = LinearRegression()
model.fit(X=train_x, y=train_y)

predictions = model.predict(X=test_x)

rmse = np.sqrt(np.mean((predictions - test_y) ** 2))
r2 = r2_score(y_true=test_y, y_pred=predictions)

print(f'\nRMSE: {rmse}\nR2: {r2}')

plt.hist(Y, bins=100, color='skyblue', edgecolor='black')
plt.title('Гистограмма распределения целевой переменной')
plt.xlabel('Значение целевой переменной')
plt.ylabel('Частота')
plt.show()

make_boxplot(param=Y, title='Коробка с усами для целевой переменной')
print('\nДа, выбросов много')

correlations = df.corr()
bad_correlations = correlations.loc['MedHouseVal'][abs(correlations.loc['MedHouseVal']) <= 0.1]
good_correlations = correlations.loc['MedHouseVal'][abs(correlations.loc['MedHouseVal']) > 0.1].drop('MedHouseVal')

print(f'\nПлохие корреляции: \n{bad_correlations}')
print(f'\nХорошие корреляции: \n{good_correlations}')

mask = np.zeros_like(correlations, dtype=bool)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(10, 8))
sns.heatmap(correlations,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            cbar_kws={"shrink": .8})

plt.title('Матрица корреляций', pad=20, fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

df.drop(axis=1, inplace=True, columns=bad_correlations.index)

X = df.drop('MedHouseVal', axis=1)
Y = df['MedHouseVal']

rmse, r2 = calculate_metrics(x=X, y=Y)

print(f'\nRMSE: {rmse}\nR2: {r2}\n')

gen = (make_boxplot(param=X[i], title=f'Коробка с усами по параметру {X[i].name}') for i in X.columns)
for i in gen:
    continue

for i in df.drop('MedHouseVal', axis=1).columns:
    lower_bound, upper_bound = define_outliers(param=df[i])
    column = df[i]
    mask = (column < lower_bound) | (column > upper_bound)
    loss_percentage = round(column[mask].count() / column.count(), 2)
    df = df[~mask]
    print(f'{loss_percentage * 100}% выбросов в признаке {column.name}')


X = df.drop('MedHouseVal', axis=1)
Y = df['MedHouseVal']

rmse, r2 = calculate_metrics(x=X, y=Y)

print(f'\nRMSE: {rmse}\nR2: {r2}\n')


df['MedInc'] = np.log(df['MedInc'])
df['HouseAge'] = np.sqrt(df['HouseAge'])

X = df.drop('MedHouseVal', axis=1)
Y = df['MedHouseVal']

rmse, r2 = calculate_metrics(x=X, y=Y)

print(f'\nRMSE: {rmse}\nR2: {r2}\n')
