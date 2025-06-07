import pandas as pd
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


def model_operation(x, y, true_res, test_x):
    model = LogisticRegression()
    model.fit(x, y)

    predictions = model.predict(test_x)

    accuracy = accuracy_score(true_res, predictions)
    precision = precision_score(true_res, predictions)
    recall = recall_score(true_res, predictions)
    f1 = f1_score(true_res, predictions)

    print('\n1 - выжил, 0 - погиб')
    print(f"accuracy: {accuracy}")
    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"f1: {f1}")


def put_random(series: pd.Series, min_val: float, max_val: float) -> pd.Series:
    min_val = round(min_val, 2)
    max_val = round(max_val, 2)

    nan_mask = series.isna()
    random_values = [round(random.uniform(min_val, max_val + 0.01), 2) for _ in range(nan_mask.sum())]

    series[nan_mask] = random_values
    return series


def remove_outliers(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Убогая функция, которая была создана исключительно для сокращения длины кода.
    Очищает тестовую и обучающую выборку от выбросов в признаке Age
    :param train_df:
    :param test_df:
    :return train_df, test_df:
    """
    ages_q1 = np.quantile(sorted(train_df['Age'].to_list()), 0.25)
    ages_q3 = np.quantile(sorted(train_df['Age'].to_list()), 0.75)
    ages_iqr = ages_q3 - ages_q1

    low_iqr = ages_q1 - (1.5 * ages_iqr)
    high_iqr = ages_q3 + (1.5 * ages_iqr)

    train_df = train_df[(train_df['Age'] > low_iqr) & (train_df['Age'] < high_iqr)]
    test_df = test_df[(test_df['Age'] > low_iqr) & (test_df['Age'] < high_iqr)]

    return train_df, test_df


train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')

y_test = pd.read_csv('test_true.csv')

x_train = train_dataset[['Sex', 'Age']]
x_test = test_dataset[['Sex', 'Age']]

# Оставил пол так как в первую очередь спасали женщин.
# Оставил возраст так как в первую очередь спасали детей.

x_test_recoded = pd.get_dummies(x_test['Sex'])
x_train_recoded = pd.get_dummies(x_train['Sex'])

x_train = pd.concat([x_train, x_train_recoded], axis=1).drop('Sex', axis=1)
x_test = pd.concat([x_test, x_test_recoded], axis=1).drop('Sex', axis=1)


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

y_unprepared_train = train_dataset[train_dataset['PassengerId'].isin(x_unprepared_train['PassengerId'])]
y_unprepared_train = y_unprepared_train['Survived']

y_unprepared_test = y_test[y_test['PassengerId'].isin(x_unprepared_test['PassengerId'])]
y_unprepared_test = y_unprepared_test['Survived']

# 'Embarked' - Порт высадки. Удалено, потому что является категориальным значением
# 'Pclass' - Класс билета. Удалено, потому что является категориальным значением
# 'Sex' - Пол. Удалено, потому что является категориальным значением
# 'Name' - Имя. Удалено, потому что строка
# 'Ticket' - Билет. Удалено, потому что смесь строки и числа, где строка это код класса, который мы удалили
# 'Cabin' - Каюта. Удалено, потому что смесь строки и числа


x_train_avg_filled = x_train.copy()
x_test_avg_filled = x_test.copy()

ages = x_train_avg_filled['Age'].dropna().to_list() + x_test_avg_filled['Age'].dropna().to_list()

x_train_avg_filled['Age'] = x_train_avg_filled['Age'].fillna(sum(ages) / len(ages))
x_test_avg_filled['Age'] = x_test_avg_filled['Age'].fillna(sum(ages) / len(ages))


x_train_rand_filled = x_train.copy()
x_test_rand_filled = x_test.copy()

x_train_rand_filled['Age'] = put_random(series=x_train_rand_filled['Age'],
                                        min_val=x_train_rand_filled['Age'].min(),
                                        max_val=x_train_rand_filled['Age'].max())
x_test_rand_filled['Age'] = put_random(series=x_test_rand_filled['Age'],
                                       min_val=x_test_rand_filled['Age'].min(),
                                       max_val=x_test_rand_filled['Age'].max())


model_operation(x=x_unprepared_train,
                y=y_unprepared_train,
                true_res=y_unprepared_test,
                test_x=x_unprepared_test)

model_operation(x=x_train_avg_filled,
                y=train_dataset['Survived'],
                true_res=y_test['Survived'],
                test_x=x_test_avg_filled)

model_operation(x=x_train_rand_filled,
                y=train_dataset['Survived'],
                true_res=y_test['Survived'],
                test_x=x_test_rand_filled)

# recall 1 потому что модель ни разу не предсказала 0 - смерть
