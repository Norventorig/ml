import pandas
from sklearn.datasets import make_classification
from sklearn.feature_selection import VarianceThreshold, f_classif, SelectKBest, SelectFromModel, \
    SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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

output += f"Средняя точность до сортировки признаков: {mean_ac_before_improvement}"


corr_matrix = dataset.corr().loc[dataset.columns, 'class']
improved_dataset = dataset[corr_matrix[corr_matrix.abs() >= 0.5].index]

x = improved_dataset.drop('class', axis=1)
y = improved_dataset['class']

mean_accuracy = cross_val_score(estimator=LogisticRegression(),
                                X=x, y=y, cv=5, scoring='accuracy').mean()

output += f"\nСредняя точность после сортировки признаков основе матрицы корреляции: {mean_accuracy}"


X_temp = dataset.drop('class', axis=1)
y_temp = dataset['class']

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

output += f"\nСредняя точность после сортировки признаков на основе VarianceThreshold: {mean_accuracy}"


X_temp = dataset.drop('class', axis=1)
y_temp = dataset['class']

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


x = dataset.drop('class', axis=1)
y = dataset['class']

model = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=1)
model.fit(x.values, y)

selector = SelectFromModel(model, prefit=True, threshold='mean')

X_selected = selector.transform(x.values)
selected_features = x.columns[selector.get_support()]

improved_dataset = pandas.DataFrame(X_selected, columns=selected_features, index=x.index)
improved_dataset['class'] = y

x = improved_dataset.drop('class', axis=1)
y = improved_dataset['class']

mean_accuracy = cross_val_score(estimator=LogisticRegression(),
                                X=x, y=y, cv=5, scoring='accuracy').mean()

output += f"\nСредняя точность после сортировки признаков L1 регуляризацией: {mean_accuracy}"


x = dataset.drop('class', axis=1)
y = dataset['class']

model = RandomForestClassifier(max_depth=40)
model.fit(x, y)

selected_features = pandas.Series(data=model.feature_importances_, index=x.columns).nlargest(3).index

x = dataset[selected_features]
y = dataset['class']

mean_accuracy = cross_val_score(estimator=LogisticRegression(),
                                X=x, y=y, cv=5, scoring='accuracy').mean()

output += f"\nСредняя точность после сортировки признаков RFC: {mean_accuracy}"


x = dataset.drop('class', axis=1)
y = dataset['class']

selector = SequentialFeatureSelector(estimator=LogisticRegression(), direction='backward',
                                     scoring='accuracy', n_features_to_select=3)

selector.fit(x, y)

x = dataset[dataset.drop('class', axis=1).columns[selector.get_support()]]
y = dataset['class']

mean_accuracy = cross_val_score(estimator=LogisticRegression(),
                                X=x, y=y, cv=5, scoring='accuracy').mean()

output += f"\nСредняя точность после сортировки признаков SequentialFeatureSelector: {mean_accuracy}"


print(output)
