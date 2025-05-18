import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def roc_curve_selfmade(y_true: pd.Series, y_score):
    thresholds_list = np.sort(np.unique(np.asarray(y_score)))[::-1]
    thresholds_list = np.insert(thresholds_list, 0, thresholds_list[0] + 1)
    thresholds_list = np.insert(thresholds_list, len(thresholds_list), thresholds_list[-1] - 1)

    y_true = y_true.to_numpy()
    predicted = {i_threshold: [int(i_score > i_threshold) for i_score in y_score] for i_threshold in thresholds_list}

    fpr_list = [(fp := len([True for index in range(len(y_true))
                               if i_predicted[index] == 1 != y_true[index]]))

                   / (fp +

                   (tn := len([True for index in range(len(y_true))
                               if i_predicted[index] == 0 == y_true[index]])))

                   for i_predicted in predicted.values()]

    tpr_list = [(tp := len([True for index in range(len(y_true))
                               if i_predicted[index] == 1 == y_true[index]]))

                   / (tp +

                   (fn := len([True for index in range(len(y_true))
                               if i_predicted[index] == 0 != y_true[index]])))

                   for i_predicted in predicted.values()]

    return fpr_list, tpr_list, thresholds_list


scaler = StandardScaler()
dataset = pd.read_csv('dataset.csv')

dataset = dataset[['height', 'weight', 'sex']]
dataset = dataset.dropna()

dataset['sex'] = (dataset['sex'] == 'male').astype(int)

x = dataset[['height', 'weight']]
y = dataset['sex']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = LogisticRegression()
model.fit(x_train, y_train)

scores = [i_score[1] for i_score in model.predict_proba(x_test)]

fpr, tpr, thresholds = roc_curve(y_test, scores)
roc_auc = roc_auc_score(y_test, scores)

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.show()

my_fpr, my_tpr, my_thresholds = roc_curve_selfmade(y_test, scores)

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve (my function)')
plt.show()

plt.figure()
plt.plot(fpr, tpr, label='Sklearn ROC (AUC = %0.2f)' % roc_auc)
plt.plot(my_fpr, my_tpr, label='Self-made ROC', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Comparison of ROC Curves')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
