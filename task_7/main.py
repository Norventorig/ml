import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import BaggingClassifier


dataset = pd.read_csv('dataset.csv')


dataset = pd.get_dummies(data=dataset, columns=['ExerciseAngina', 'Sex'], drop_first=True)
dataset = pd.get_dummies(data=dataset)

dataset = dataset.rename(columns={'ExerciseAngina_Y': 'ExerciseAngina', 'Sex_M': 'Male'})


X = dataset.drop('HeartDisease', axis=1)
Y = dataset['HeartDisease']

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)


dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier()
bc = BaggingClassifier(n_estimators=15, max_samples=0.8)

dtc.fit(X=train_x, y=train_y)
rfc.fit(X=train_x, y=train_y)
bc.fit(train_x, train_y)

dtc_rep = classification_report(y_true=test_y, y_pred=dtc.predict(test_x))
rfc_rep = classification_report(y_true=test_y, y_pred=rfc.predict(test_x))
bc_rep = classification_report(y_true=test_y, y_pred=bc.predict(test_x))

print(f"\nМетрики BaggingClassifier:\n{bc_rep}")
print(f"\nМетрики RandomForestClassifier:\n{rfc_rep}")
print(f"\nМетрики DecisionTreeClassifier:\n{dtc_rep}")


feature_importance_df = pd.DataFrame(data={'Importance': rfc.feature_importances_, 'Column': train_x.columns})
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
bars = plt.barh(
    y=feature_importance_df['Column'],
    width=feature_importance_df['Importance'],
    color='skyblue'
)

plt.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)

plt.xlabel('Importance')
plt.title('Feature Importance (Random Forest)')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()

plt.show()
