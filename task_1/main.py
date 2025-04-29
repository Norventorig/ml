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


x = pd.DataFrame(data={'age': df['age'], 'income': le.fit(df['income']), 'race': le.fit(df['race'])})
y = pd.Series(data=le.fit(df['gender']))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

log_model = LogisticRegression()
log_model.fit(x_train, y_train)
predicted = log_model.predict(x_test)

print(predicted.head())
