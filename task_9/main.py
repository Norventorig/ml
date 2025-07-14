import pandas
from sklearn.datasets import make_classification
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


x_data_generated, y_data_generated = make_classification(scale=1)

x_data = pandas.DataFrame(data=x_data_generated, columns=[f'col_{i}' for i in range(1, len(x_data_generated[0]) + 1)])
y_data = pandas.Series(data=y_data_generated, name='class')

dataset = pandas.concat([x_data, y_data], axis=1)


corr_matrix = dataset.corr().loc[dataset.columns, 'class']
improved_dataset = dataset[corr_matrix[corr_matrix.abs() >= 0.5].index]

X_temp = improved_dataset.drop('class', axis=1)
y_temp = improved_dataset['class']

var_thr = VarianceThreshold(threshold=0.1)
X_temp_selected = var_thr.fit_transform(X_temp)

selected_columns = X_temp.columns[var_thr.get_support()]
improved_dataset = pandas.DataFrame(
    X_temp_selected,
    columns=selected_columns,
    index=X_temp.index
)
improved_dataset['class'] = y_temp


x_bi = dataset.drop('class', axis=1)
y_bi = dataset['class']

x_ai = improved_dataset.drop('class', axis=1)
y_ai = improved_dataset['class']

mean_ac_before_improvement = cross_val_score(estimator=LogisticRegression(),
                                             X=x_bi, y=y_bi, cv=5, scoring='accuracy').mean()
mean_ac_after_improvement = cross_val_score(estimator=LogisticRegression(),
                                             X=x_ai, y=y_ai, cv=5, scoring='accuracy').mean()

print(f"Средняя точность до сортировки признаков: {mean_ac_before_improvement}"
      f"\nСредняя точность после сортировки признаков: {mean_ac_after_improvement}")
