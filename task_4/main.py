import pandas as pd
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')

x_unprepared_train = train_dataset.copy()
x_unprepared_train.dropna(inplace=True)
x_unprepared_train.drop(['Embarked', 'Pclass', 'Sex', 'Name', 'Ticket', 'Cabin', 'Survived'], axis=1, inplace=True)

x_unprepared_test = test_dataset.copy()
x_unprepared_test.dropna(inplace=True)
x_unprepared_test.drop(['Embarked', 'Pclass', 'Sex', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

y_train = train_dataset['Survived']

y_test = pd.read_csv('test_true.csv')
y_test = y_test.merge(test_dataset['PassengerId'], how='inner', on='PassengerId')

# 'Embarked' - Порт высадки. Удалено, потому что является категориальным значением
# 'Pclass' - Класс билета. Удалено, потому что является категориальным значением
# 'Sex' - Пол. Удалено, потому что является категориальным значением
# 'Name' - Имя. Удалено, потому что строка
# 'Ticket' - Билет. Удалено, потому что смесь строки и числа, где строка это код класса, который мы удалили
# 'Cabin' - Каюта. Удалено, потому что смесь строки и числа

model = LogisticRegression()
model.fit(x_unprepared_train, y_train)

predictions = model.predict(x_unprepared_test)

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


x_train = train_dataset[['Sex', 'Age']]
x_test = test_dataset[['Sex', 'Age']]

# Оставил пол так как в первую очередь спасали женщин.
# Оставил возраст так как в первую очередь спасали детей.

los_percentage_train = len(x_train.dropna()) * 100 / len(x_train)
los_percentage_test = len(x_test.dropna()) * 100 / len(x_test)

print(f'Процент данных, который будет потерян, если просто удалить пропуски:')
print(f'Test - {los_percentage_test}')
print(f'Train - {los_percentage_train}')

ages = x_train['Age'].to_list() + x_test['Age'].to_list()

x_train_avg_filled = x_train['Age'].fillna(sum(ages) / len(ages))
x_test_avg_filled = x_test['Age'].fillna(sum(ages) / len(ages))


def put_random(series: pd.Series, min_val: float, max_val: float):
    min_val = round(min_val, 2)
    max_val = round(max_val, 2)

    nan_mask = series.isna()
    random_values = [round(random.uniform(min_val, max_val + 0.01), 2) for _ in range(nan_mask.sum())]

    series[nan_mask] = random_values
    return series


x_train_rand_filled = put_random(series=x_train['Age'], min_val=x_train['Age'].min(), max_val=x_train['Age'].max())
x_test_rand_filled = put_random(series=x_test['Age'], min_val=x_test['Age'].min(), max_val=x_test['Age'].max())
