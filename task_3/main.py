import pandas as pd


dataset = pd.read_csv('dataset.csv')

dataset['nationality'] = dataset['nationality'].astype('category')

dataset['sex'] = dataset['sex'] == 'male'
dataset['sex'] = dataset['sex'].astype(bool)

dataset['dob'] = pd.to_datetime(dataset['dob'], format='%m/%d/%y')

dataset['sport'] = dataset['sport'].astype('category')

dataset['gold'] = dataset['gold'].astype(bool)
dataset['silver'] = dataset['silver'].astype(bool)
dataset['bronze'] = dataset['bronze'].astype(bool)
