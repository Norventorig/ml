from sklearn.datasets import fetch_california_housing
import pandas as pd


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

print('Пропуски есть' if 0 != sum(df.isnull().sum().to_list()) else 'Пропусков нет')
