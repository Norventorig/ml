import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


pd.set_option('display.max_rows', None)  # Все строки
pd.set_option('display.max_columns', None)  # Все столбцы
pd.set_option('display.width', None)  # Автоперенос отключен
pd.set_option('display.max_colwidth', None)  # Полное содержимое ячеек


train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')
test_y = pd.read_csv('test_true.csv')

train_dataset.dropna(inplace=True)
train_dataset.drop(['Embarked', 'Pclass', 'Sex', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

x_unprepared_train = train_dataset.drop('Survived', axis=1)
train_y = train_dataset['Survived']

test_dataset.dropna(inplace=True)
test_dataset.drop(['Embarked', 'Pclass', 'Sex', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

x_unprepared_test = test_dataset
test_y = test_y.merge(test_dataset['PassengerId'], how='inner', on='PassengerId')

# 'Embarked' - Порт высадки. Удалено, потому что является категориальным значением
# 'Pclass' - Класс билета. Удалено, потому что является категориальным значением
# 'Sex' - Пол. Удалено, потому что является категориальным значением
# 'Name' - Имя. Удалено, потому что строка
# 'Ticket' - Билет. Удалено, потому что смесь строки и числа, где строка это код класса, который мы удалили
# 'Cabin' - Каюта. Удалено, потому что смесь строки и числа

model = LogisticRegression()
model.fit(x_unprepared_train, train_y)

predictions = model.predict(x_unprepared_test)

accuracy = accuracy_score(test_y['Survived'], predictions)
precision = precision_score(test_y['Survived'], predictions)
recall = recall_score(test_y['Survived'], predictions)
f1 = f1_score(test_y['Survived'], predictions)

print('1 - выжил, 0 - погиб')
print(f"accuracy: {accuracy}")
print(f"precision: {precision}")
print(f"recall: {recall}")
print(f"f1: {f1}")

# recall 1 потому что модель ни разу не предсказала 0 - смерть
