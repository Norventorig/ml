from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report


def make_classification_report(x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    prediction = model.predict(X=x_test)

    return classification_report(y_pred=prediction, y_true=y_test)


dataset = fetch_ucirepo(id=40).data.features

# data_graph = pd.Series({'Catholic': dataset['religion'][dataset['religion'] == 0].count(),
#                         'Other Christian': dataset['religion'][dataset['religion'] == 1].count(),
#                         'Muslim': dataset['religion'][dataset['religion'] == 2].count(),
#                         'Buddhist': dataset['religion'][dataset['religion'] == 3].count(),
#                         'Hindu': dataset['religion'][dataset['religion'] == 4].count(),
#                         'Ethnic': dataset['religion'][dataset['religion'] == 5].count(),
#                         'Marxist': dataset['religion'][dataset['religion'] == 6].count(),
#                         'Others': dataset['religion'][dataset['religion'] == 7].count()})
#
# plt.pie(x=data_graph, labels=data_graph.index)
# plt.title("distribution of religions")
# plt.show()

# data_graph = pd.crosstab(
#     dataset['religion'].replace({
#         0: 'Catholic',
#         1: 'Other Christian',
#         2: 'Muslim',
#         3: 'Buddhist',
#         4: 'Hindu',
#         5: 'Ethnic',
#         6: 'Marxist',
#         7: 'Others'
#     }),
#     dataset['zone'].replace({
#         1: 'NE',
#         2: 'SE',
#         3: 'SW',
#         4: 'NW'
#     }))
#
# sns.heatmap(data=data_graph, annot=True)
# plt.title('Распределение религий по зонам')
# plt.xlabel('Религия')
# plt.ylabel('Географическая зона')
# plt.show()


dataset = pd.get_dummies(data=dataset, columns=['zone', 'landmass', 'language', 'mainhue', 'botright', 'topleft'])
dataset.drop('name', axis=1, inplace=True)

X = dataset.drop('religion', axis=1)
y = dataset['religion']

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)


model = RandomForestClassifier()

print(f'Классификация при оригинальных данных: '
      f'\n{make_classification_report(x_train=train_x, y_train=train_y, x_test=test_x, y_test=test_y)}')


# data_graph = dataset.corr()['religion'].abs().sort_values(ascending=False).head(6).index
# data_graph = dataset[data_graph].corr()
#
# sns.heatmap(data=data_graph, annot=True)
# plt.title('Correlation matrix')
# plt.show()


# for i_param in X.columns:
#     sns.boxplot(data=X[i_param], orient='h')
#     plt.title(f'{i_param} boxplot')
#     plt.show()


standart_scaler = StandardScaler()
X['area'] = standart_scaler.fit_transform(X[['area']])

print(f'Классификация после нормализации area: '
      f'\n{make_classification_report(x_train=train_x, y_train=train_y, x_test=test_x, y_test=test_y)}')


ros = RandomOverSampler(sampling_strategy={i: 35 if y.value_counts()[i] < 35 else y.value_counts()[i]
                                           for i in y.value_counts().index.to_list()})
X, y = ros.fit_resample(X=X, y=y)

rus = RandomUnderSampler(sampling_strategy={i: 35 if y.value_counts()[i] > 35 else y.value_counts()[i]
                                            for i in y.value_counts().index.to_list()})
X, y = rus.fit_resample(X=X, y=y)

print(f'Классификация после oversampling/undersampling: '
      f'\n{make_classification_report(x_train=train_x, y_train=train_y, x_test=test_x, y_test=test_y)}')


X = pd.DataFrame(data=PCA(n_components=5).fit_transform(X))

print(f'Классификация после уменьшения размерности пространства признаков: '
      f'\n{make_classification_report(x_train=train_x, y_train=train_y, x_test=test_x, y_test=test_y)}')


mi = mutual_info_classif(X=X, y=y)
print(f'Mutual Information признаков с уменьшенной размерностью: {mi}')
