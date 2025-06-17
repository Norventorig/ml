from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


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
