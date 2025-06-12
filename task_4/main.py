import pandas as pd
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def model_operation(train_x, train_y, test_x, test_y):
    model = LogisticRegression()
    model.fit(train_x, train_y)

    predictions = model.predict(test_x)

    accuracy = accuracy_score(test_y, predictions)
    precision = precision_score(test_y, predictions)
    recall = recall_score(test_y, predictions)
    f1 = f1_score(test_y, predictions)

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


def remove_outliers(train_df: pd.DataFrame, test_df: pd.DataFrame, outlier_param: str):
    """
    Убогая функция, которая была создана исключительно для сокращения длины кода.
    Очищает тестовую и обучающую выборку от выбросов по признаку
    :param train_df:
    :param test_df:
    :param outlier_param:
    :return train_df, test_df:
    """
    ages_q1 = np.quantile(sorted(train_df[outlier_param].to_list()), 0.25)
    ages_q3 = np.quantile(sorted(train_df[outlier_param].to_list()), 0.75)
    ages_iqr = ages_q3 - ages_q1

    low_iqr = ages_q1 - (1.5 * ages_iqr)
    high_iqr = ages_q3 + (1.5 * ages_iqr)

    train_df = train_df[(train_df[outlier_param] > low_iqr) & (train_df[outlier_param] < high_iqr)]
    test_df = test_df[(test_df[outlier_param] > low_iqr) & (test_df[outlier_param] < high_iqr)]

    return train_df, test_df


orig_dataset = pd.read_csv('train.csv')


orig_dataset_copy = orig_dataset[['Sex', 'Age', 'Survived']]

x = orig_dataset_copy.drop('Survived', axis=1)
y = orig_dataset_copy['Survived']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Оставил пол так как в первую очередь спасали женщин.
# Оставил возраст так как в первую очередь спасали детей.


x_test_recoded = pd.get_dummies(x_test['Sex'])
x_train_recoded = pd.get_dummies(x_train['Sex'])

x_train = pd.concat([x_train, x_train_recoded], axis=1).drop(['Sex', 'female'], axis=1)
x_test = pd.concat([x_test, x_test_recoded], axis=1).drop(['Sex', 'female'], axis=1)


los_percentage_train = len(x_train.dropna()) * 100 / len(x_train)
los_percentage_test = len(x_test.dropna()) * 100 / len(x_test)

print(f'Процент данных, который будет потерян, если просто удалить пропуски:')
print(f'Test - {los_percentage_test}')
print(f'Train - {los_percentage_train}')


orig_dataset_copy = orig_dataset.drop(['Embarked', 'Pclass', 'Sex', 'Name', 'Ticket', 'Cabin'], axis=1).dropna()

x = orig_dataset_copy.drop('Survived', axis=1)
y = orig_dataset_copy['Survived']

x_unprepared_train, x_unprepared_test, y_unprepared_train, y_unprepared_test = train_test_split(x, y, test_size=0.3)

# 'Embarked' - Порт высадки. Удалено, потому что является категориальным значением
# 'Pclass' - Класс билета. Удалено, потому что является категориальным значением
# 'Sex' - Пол. Удалено, потому что является категориальным значением
# 'Name' - Имя. Удалено, потому что строка
# 'Ticket' - Билет. Удалено, потому что смесь строки и числа, где строка это код класса, который мы удалили
# 'Cabin' - Каюта. Удалено, потому что смесь строки и числа


x_train_avg_filled = x_train.copy()
x_test_avg_filled = x_test.copy()

ages = x_train_avg_filled['Age'].dropna().to_list()
avg_age = sum(ages) / len(ages)

x_train_avg_filled['Age'] = x_train_avg_filled['Age'].fillna(avg_age)
x_test_avg_filled['Age'] = x_test_avg_filled['Age'].fillna(avg_age)


x_train_rand_filled = x_train.copy()
x_test_rand_filled = x_test.copy()

x_train_rand_filled['Age'] = put_random(series=x_train_rand_filled['Age'],
                                        min_val=x_train_rand_filled['Age'].min(),
                                        max_val=x_train_rand_filled['Age'].max())
x_test_rand_filled['Age'] = put_random(series=x_test_rand_filled['Age'],
                                        min_val=x_train_rand_filled['Age'].min(),
                                        max_val=x_train_rand_filled['Age'].max())


model_operation(train_x=x_unprepared_train,
                train_y=y_unprepared_train,
                test_y=y_unprepared_test,
                test_x=x_unprepared_test)

model_operation(train_x=x_train_avg_filled,
                train_y=y_train,
                test_y=y_test,
                test_x=x_test_avg_filled)

model_operation(train_x=x_train_rand_filled,
                train_y=y_train,
                test_y=y_test,
                test_x=x_test_rand_filled)


x_train_avg_filled_without_outliers, x_test_avg_filled_without_outliers = remove_outliers(
    train_df=x_train_avg_filled,
    test_df=x_test_avg_filled,
    outlier_param='Age')

x_train_rand_filled_without_outliers, x_test_rand_filled_without_outliers = remove_outliers(
    train_df=x_train_rand_filled,
    test_df=x_test_rand_filled,
    outlier_param='Age')

model_operation(train_x=x_train_avg_filled_without_outliers,
                train_y=orig_dataset[orig_dataset.index.isin(x_train_avg_filled_without_outliers.index)]['Survived'],
                test_y=orig_dataset[orig_dataset.index.isin(x_test_avg_filled_without_outliers.index)]['Survived'],
                test_x=x_test_avg_filled_without_outliers)

model_operation(train_x=x_train_rand_filled_without_outliers,
                train_y=orig_dataset[orig_dataset.index.isin(x_train_rand_filled_without_outliers.index)]['Survived'],
                test_y=orig_dataset[orig_dataset.index.isin(x_test_rand_filled_without_outliers.index)]['Survived'],
                test_x=x_test_rand_filled_without_outliers)
