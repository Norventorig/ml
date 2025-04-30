import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


le = LabelEncoder()
df = pd.read_csv('dataset.csv')

df['income'] = df['income'].astype('category')


df.groupby('income')['age'].mean().plot(kind='bar')

plt.title('mean age on income')
plt.xlabel('income')
plt.ylabel('age')
plt.show()


index = df['income'].cat.categories
males = df[df['gender'] == 'Male'].groupby('income')['gender'].count()
females = df[df['gender'] == 'Female'].groupby('income')['gender'].count()

gender_income_df = pd.DataFrame(data={'males': males, "females": females}, index=index)
gender_income_df['males'] = gender_income_df['males'].astype('float64')
gender_income_df['females'] = gender_income_df['females'].astype('float64')

all_males = gender_income_df.loc[:, 'males'].sum()
all_females = gender_income_df.loc[:, 'females'].sum()

gender_income_df.loc[[True, False], 'males'] = \
    round(100 / (all_males / gender_income_df.loc[[True, False], 'males']), 2)

gender_income_df.loc[[False, True], 'males'] = \
    round(100 / (all_males / gender_income_df.loc[[False, True], 'males']), 2)

gender_income_df.loc[[True, False], 'females'] = \
    round(100 / (all_females / gender_income_df.loc[[True, False], 'females']), 2)

gender_income_df.loc[[False, True], 'females'] = \
    round(100 / (all_females / gender_income_df.loc[[False, True], 'females']), 2)

gender_income_df.plot(kind='bar')
plt.title('gender on income')
plt.xlabel('income')
plt.ylabel('percentage')
plt.show()


x = pd.DataFrame()
le.fit(df['gender'])
x['gender'] = le.transform(df['gender'])
le.fit(df['race'])
x['race'] = le.transform(df['race'])
x['hours-per-week'] = df['hours-per-week']
x['age'] = df['age']

le.fit(df['income'])
y = pd.Series(data=le.transform(df['income']))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

log_model = LogisticRegression()
log_model.fit(x_train, y_train)

svc_model = SVC()
svc_model.fit(x_train, y_train)

log_model_score = log_model.score(x_test, y_test)
svc_model_score = svc_model.score(x_test, y_test)

print(f'\nТочность модели построенной на Логистической регрессии: {log_model_score}'
      f'\nТочность модели построенной на Методе опорных векторов: {svc_model_score}')

print('Модель построенная на Логистической регрессии лучше' if log_model_score > svc_model_score else
      ('Модель построенная на Методе опорных векторов лучше' if log_model_score < svc_model_score else
       'Точность моделей идентична'))
