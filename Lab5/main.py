import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#загрузка и выбор 100 строк датасета
dataset = pd.read_csv(r"/Users/Игорь/Desktop/Data_COVID_Tests.csv")
dataset.sample(frac=1)
dataset.drop('DATE_DIED', axis = 1, inplace = True)
dataset = dataset.head(100)

#выделение целевого столбца и выбор первых 100 строк
aim_label = dataset['CLASIFFICATION_FINAL']
aim_label = aim_label.head(100)
dataset.drop('CLASIFFICATION_FINAL', axis = 1, inplace = True)

#отбор значимых признаков
model = ExtraTreesClassifier()
model.fit(dataset, aim_label)
print("Отбор признаков по степени важности: ", model.feature_importances_)

#формирование новой выборки значимых признаков
columns = [0, 1, 2, 3, 4, 7]
dataset.drop(dataset.columns [columns], axis = 1, inplace = True)
#print("Выборка из значимых признаков")
#print(dataset.head())

#разделение датасета на обучающую и тестовую выборки
X_train, X_test, Y_train, Y_test = train_test_split(dataset, aim_label, test_size=0.1, random_state=0)
#print("Обучающая выборка")
#print(X_train)
#print("Тестовая выборка")
#print(X_test)

#обучение модели
sgd = SGDClassifier (loss='hinge', penalty='l2', alpha=1e-3, random_state=0, max_iter=5, tol=None)
sgd.fit(X_train, Y_train)
y_pred = sgd.predict(X_test)
score = accuracy_score(Y_test, y_pred)
print(f'Точность SGD-классификатора: {round(score * 100, 2)}%')

svc = LinearSVC()
svc.fit(X_train, Y_train)
y_pred = svc.predict(X_test)
score = accuracy_score(Y_test, y_pred)
print(f'Точность SVC-классификатора: {round(score * 100, 2)}%')

rf = RandomForestClassifier(max_depth = 2, random_state = 0)
rf.fit(X_train, Y_train)
y_pred = rf.predict(X_test)
score = accuracy_score(Y_test, y_pred)
print(f'Точность случайного леса: {round(score * 100, 2)}%')

#очищенные данные
#очистка выборки значимых признаков
dataset.loc[dataset['time'] > 2, 'time'] = 3
dataset.loc[dataset['date'] > 2, 'date'] = 3
dataset.loc[dataset['confirmed'] > 2, 'confirmed'] = 3
dataset.loc[dataset['test'] > 2, 'test'] = 3
dataset = dataset.fillna(3)

#разделение датасета на обучающую и тестовую выборки
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(dataset, aim_label, test_size=0.1, random_state=0)

#обучение модели
sgd = SGDClassifier (loss='hinge', penalty='l2', alpha=1e-3, random_state=0, max_iter=5, tol=None)
sgd.fit(X_train1, Y_train1)
y_pred1 = sgd.predict(X_test1)
score = accuracy_score(Y_test1, y_pred1)
print(f'Точность SGD-классификатора: {round(score * 100, 2)}%')

svc = LinearSVC()
svc.fit(X_train1, Y_train1)
y_pred1= svc.predict(X_test1)
score = accuracy_score(Y_test1, y_pred1)
print(f'Точность SVC-классификатора: {round(score * 100, 2)}%')

rf = RandomForestClassifier(max_depth = 2, random_state = 0)
rf.fit(X_train1, Y_train1)
y_pred1 = rf.predict(X_test1)
score = accuracy_score(Y_test1, y_pred1)
print(f'Точность случайного леса: {round(score * 100, 2)}%')

plt.subplot(1, 2, 1)
plt.hist(Y_test1)
plt.subplot(1, 2, 2)
plt.hist(y_pred1)
plt.show()
