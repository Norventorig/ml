import pandas as pd
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


def put_random(series: pd.Series, min_val: float, max_val: float):
    min_val = round(min_val, 2)
    max_val = round(max_val, 2)

    nan_mask = series.isna()
    random_values = [round(random.uniform(min_val, max_val + 0.01), 2) for _ in range(nan_mask.sum())]

    series[nan_mask] = random_values
    return series


train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')

x_train = train_dataset[['Sex', 'Age']]
x_test = test_dataset[['Sex', 'Age']]

# Оставил пол так как в первую очередь спасали женщин.
# Оставил возраст так как в первую очередь спасали детей.

x_test_recoded = pd.get_dummies(x_test['Sex'])
x_train_recoded = pd.get_dummies(x_train['Sex'])

x_train = pd.concat([x_train, x_train_recoded], axis=1).drop('Sex', axis=1)
x_test = pd.concat([x_test, x_test_recoded], axis=1).drop('Sex', axis=1)

y_train = train_dataset['Survived']

y_test = pd.read_csv('test_true.csv')
y_test = y_test.merge(test_dataset['PassengerId'], how='inner', on='PassengerId')


los_percentage_train = len(x_train.dropna()) * 100 / len(x_train)
los_percentage_test = len(x_test.dropna()) * 100 / len(x_test)

print(f'Процент данных, который будет потерян, если просто удалить пропуски:')
print(f'Test - {los_percentage_test}')
print(f'Train - {los_percentage_train}')


x_unprepared_train = train_dataset.copy()
x_unprepared_train.dropna(inplace=True)
x_unprepared_train.drop(['Embarked', 'Pclass', 'Sex', 'Name', 'Ticket', 'Cabin', 'Survived'], axis=1, inplace=True)

x_unprepared_test = test_dataset.copy()
x_unprepared_test.dropna(inplace=True)
x_unprepared_test.drop(['Embarked', 'Pclass', 'Sex', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# 'Embarked' - Порт высадки. Удалено, потому что является категориальным значением
# 'Pclass' - Класс билета. Удалено, потому что является категориальным значением
# 'Sex' - Пол. Удалено, потому что является категориальным значением
# 'Name' - Имя. Удалено, потому что строка
# 'Ticket' - Билет. Удалено, потому что смесь строки и числа, где строка это код класса, который мы удалили
# 'Cabin' - Каюта. Удалено, потому что смесь строки и числа


ages = x_train['Age'].to_list() + x_test['Age'].to_list()

x_train_avg_filled = x_train['Age'].fillna(sum(ages) / len(ages))
x_test_avg_filled = x_test['Age'].fillna(sum(ages) / len(ages))


x_train_rand_filled = put_random(series=x_train['Age'], min_val=x_train['Age'].min(), max_val=x_train['Age'].max())
x_test_rand_filled = put_random(series=x_test['Age'], min_val=x_test['Age'].min(), max_val=x_test['Age'].max())


unprepared_model = LogisticRegression()
unprepared_model.fit(x_unprepared_train, y_train)

predictions = unprepared_model.predict(x_unprepared_test)

accuracy = accuracy_score(y_test['Survived'], predictions)
precision = precision_score(y_test['Survived'], predictions)
recall = recall_score(y_test['Survived'], predictions)
f1 = f1_score(y_test['Survived'], predictions)

print('1 - выжил, 0 - погиб')
print(f"accuracy: {accuracy}")
print(f"precision: {precision}")
print(f"recall: {recall}")
print(f"f1: {f1}")

# recall 1 потому что модель ни разу не предсказала 0 - смерть


avg_filled_model = LogisticRegression()
avg_filled_model.fit(x_train_avg_filled, y_train)

predictions = avg_filled_model.predict(x_test_avg_filled)

accuracy = accuracy_score(y_test['Survived'], predictions)
precision = precision_score(y_test['Survived'], predictions)
recall = recall_score(y_test['Survived'], predictions)
f1 = f1_score(y_test['Survived'], predictions)

print('1 - выжил, 0 - погиб')
print(f"accuracy: {accuracy}")
print(f"precision: {precision}")
print(f"recall: {recall}")
print(f"f1: {f1}")


rand_filled_model = LogisticRegression()
rand_filled_model.fit(x_train_rand_filled, y_train)

predictions = rand_filled_model.predict(x_test_rand_filled)

accuracy = accuracy_score(y_test['Survived'], predictions)
precision = precision_score(y_test['Survived'], predictions)
recall = recall_score(y_test['Survived'], predictions)
f1 = f1_score(y_test['Survived'], predictions)

print('1 - выжил, 0 - погиб')
print(f"accuracy: {accuracy}")
print(f"precision: {precision}")
print(f"recall: {recall}")
print(f"f1: {f1}")


# ages_q1 = np.quantile(sorted(x_train['Age'].to_list()), 0.25)
# ages_q3 = np.quantile(sorted(x_train['Age'].to_list()), 0.75)
# ages_iqr = ages_q3 - ages_q1
#
# x_train_without_outlier = x_train[[x_train['Ages'] in range(ages_q1 - (1.5 * ages_iqr), ages_q3 + (1.5 * ages_iqr))]]
# # x_test_without_outlier =
