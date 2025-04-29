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

pd.DataFrame(data={'males': males, "females": females}, index=index).plot(kind='bar')

plt.title('gender on income')
plt.xlabel('income')
plt.ylabel('amount')
plt.show()


x = pd.DataFrame()
le.fit(df['gender'])
x['gender'] = le.transform(df['gender'])
le.fit(df['race'])
x['race'] = le.transform(df['race'])

le.fit(df['income'])
y = pd.Series(data=le.transform(df['income']))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

log_model = LogisticRegression()
log_model.fit(x_train, y_train)

svc_model = SVC(kernel='linear')
svc_model.fit(x_train, y_train)
