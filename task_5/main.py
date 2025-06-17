from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split

# :Attribute Information:
#     - MedInc        median income in block group
#     - HouseAge      median house age in block group
#     - AveRooms      average number of rooms per household
#     - AveBedrms     average number of bedrooms per household
#     - Population    block group population
#     - AveOccup      average number of household members
#     - Latitude      block group latitude
#     - Longitude     block group longitude


df = pd.DataFrame(data=fetch_california_housing().data)


null_count = sum(df.isnull().sum().to_list())
print('Пропуски есть' if 0 != null_count else 'Пропусков нет')
df.dropna(axis=1, inplace=True)

X = df[['HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']]
Y = df['MedInc']

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=1)
