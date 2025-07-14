import pandas
from sklearn.datasets import make_classification
from sklearn.feature_selection import VarianceThreshold, f_classif, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


output = str()


x_data_generated, y_data_generated = make_classification(scale=1)

x_data = pandas.DataFrame(data=x_data_generated, columns=[f'col_{i}' for i in range(1, len(x_data_generated[0]) + 1)])
y_data = pandas.Series(data=y_data_generated, name='class')

dataset = pandas.concat([x_data, y_data], axis=1)


x = dataset.drop('class', axis=1)
y = dataset['class']

mean_ac_before_improvement = cross_val_score(estimator=LogisticRegression(),
                                             X=x, y=y, cv=5, scoring='accuracy').mean()

output += f"\nСредняя точность до сортировки признаков: {mean_ac_before_improvement}"


corr_matrix = dataset.corr().loc[dataset.columns, 'class']
improved_dataset = dataset[corr_matrix[corr_matrix.abs() >= 0.5].index]

X_temp = improved_dataset.drop('class', axis=1)
y_temp = improved_dataset['class']

selector = VarianceThreshold(threshold=0.1)
X_temp_selected = selector.fit_transform(X_temp)

selected_columns = X_temp.columns[selector.get_support()]
improved_dataset = pandas.DataFrame(
    X_temp_selected,
    columns=selected_columns,
    index=X_temp.index
)
improved_dataset['class'] = y_temp

x = improved_dataset.drop('class', axis=1)
y = improved_dataset['class']

mean_accuracy = cross_val_score(estimator=LogisticRegression(),
                                X=x, y=y, cv=5, scoring='accuracy').mean()

output += f"\nСредняя точность после сортировки признаков основе матрицы корреляции: {mean_accuracy}"


improved_dataset = dataset.copy()

X_temp = improved_dataset.drop('class', axis=1)
y_temp = improved_dataset['class']

selector = SelectKBest(score_func=f_classif, k=5)

X_temp_selected = selector.fit_transform(X_temp, y_temp)

selected_columns = X_temp.columns[selector.get_support()]

improved_dataset = pandas.DataFrame(
    X_temp_selected,
    columns=selected_columns,
    index=X_temp.index
)
improved_dataset['class'] = y_temp

x = improved_dataset.drop('class', axis=1)
y = improved_dataset['class']

mean_accuracy = cross_val_score(estimator=LogisticRegression(),
                                X=x, y=y, cv=5, scoring='accuracy').mean()

output += f"\nСредняя точность после сортировки признаков функцией SelectKBest через f_classif: {mean_accuracy}"


print(output)
