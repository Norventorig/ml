import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.linear_model import LogisticRegression


dataset = pd.read_csv('dataset.csv')
dataset = pd.get_dummies(columns=dataset.drop('HeartDisease', axis=1).columns.to_list(), data=dataset)


X = dataset.drop('HeartDisease', axis=1)
y = dataset['HeartDisease']

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=1)


cv_results = cross_validate(estimator=LogisticRegression(), y=y, X=X, cv=10,
                            scoring=['accuracy', 'recall', 'precision', 'f1'])


grid_params = {'C': [0.1, 0.5, 1, 10, 100],
               'solver': ['liblinear', 'lbfgs', 'saga', 'newton-cg'],
               'penalty': ['l2', None],
               'max_iter': [10000, 20000, 50000, 100000]}
grid_search = GridSearchCV(estimator=LogisticRegression(), param_grid=grid_params, scoring='accuracy')

grid_search.fit(X=train_x, y=train_y)
gs_params = grid_search.best_params_
gs_score = grid_search.best_score_

