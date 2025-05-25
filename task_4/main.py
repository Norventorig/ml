import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder


train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')

train_dataset.dropna(inplace=True)
train_dataset.drop(['Embarked', 'Pclass', 'Sex', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

x_unprepared_train = train_dataset.drop('Survived', axis=1)
y_train = train_dataset['Survived']

test_dataset.dropna(inplace=True)
test_dataset.drop(['Embarked', 'Pclass', 'Sex', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

x_unprepared_test = test_dataset
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


x_prepared_train = train_dataset[['Sex', 'Age']]
x_prepared_test = test_dataset[['Sex', 'Age']]

x_prepared_train.dropna(inplace=True)
x_prepared_test.dropna(inplace=True)

encoder = OneHotEncoder(sparse_output=False)
encoder.fit(x_prepared_train[["Sex"]])

encoded_sex_train = pd.DataFrame(data=encoder.transform(x_prepared_train[["Sex"]]),
                                 columns=['Female', 'Male'])
encoded_sex_test = pd.DataFrame(data=encoder.transform(x_prepared_test[["Sex"]]),
                                columns=['Female', 'Male'])

x_prepared_train = pd.concat([x_prepared_train.drop("Sex", axis=1), encoded_sex_train], axis=1)
x_prepared_test = pd.concat([x_prepared_test.drop("Sex", axis=1), encoded_sex_test], axis=1)


# y_train
# y_test
