from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def define_outliers(param):
    """
    Функция рассчитывающая нижнюю и верхнюю границы выбросов
    :param param: 
    :return lower, upper: 
    """
    q1 = np.quantile(param, 0.25)
    q3 = np.quantile(param, 0.75)

    iqr = q3 - q1

    lower = q1 - iqr * 1.5
    upper = q3 + iqr * 1.5

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

lower_bound, upper_bound = define_outliers(param=predictions)

plt.hist(predictions, bins=100, color='skyblue', edgecolor='black')
plt.title('Гистограмма распределения целевой переменной')
plt.xlabel('Значение целевой переменной')
plt.ylabel('Частота')
plt.show()

plt.figure(figsize=(10, 6))
plt.boxplot(predictions, vert=False, patch_artist=True)

plt.axvline(lower_bound, color='r', linestyle='--', alpha=0.7, label=f'Lower bound: {lower_bound:.2f}')
plt.axvline(upper_bound, color='g', linestyle='--', alpha=0.7, label=f'Upper bound: {upper_bound:.2f}')

plt.title('Коробка с усами для предсказаний модели', fontsize=14)
plt.xlabel('Предсказанные значения', fontsize=12)
plt.grid(axis='x', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

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

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=1)

model.fit(X=train_x, y=train_y)

predictions = model.predict(X=test_x)

rmse = np.sqrt(np.mean((predictions - test_y) ** 2))
r2 = r2_score(y_true=test_y, y_pred=predictions)

print(f'\nRMSE: {rmse}\nR2: {r2}')