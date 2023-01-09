import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn import tree
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#загрузка и выбор 100 строк датасета
dataset = pd.read_csv(r"/Users/Игорь/Desktop/Data_COVID_Tests.csv")
dataset.sample(frac=1)
dataset.drop('date', axis = 1, inplace = True)
dataset = dataset.head(100)

#выделение целевого столбца и выбор первых 100 строк
aim_label = dataset['deceased']
aim_label = aim_label.head(100)
dataset.drop('deceased', axis = 1, inplace = True)

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


tree_model = tree.DecisionTreeClassifier()
tree_model.fit (X_train1, Y_train1)
predictions = tree_model.predict(X_test1)
print("Accuracy: {}".format((tree_model.score(X_test1,Y_test1))*100))

svm_model = svm.SVC(kernel='linear')
svm_model.fit(X_train2, Y_train2)
predictions = svm_model.predict(X_test2)
print("Accuracy: {}".format((svm_model.score(X_test2,Y_test2))*100))

lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train3, Y_train3)
predictions = lda_model.predict(X_test3)
print("Accuracy: {}".format((lda_model.score(X_test3,Y_test3))*100))

#Обработка данных

scaler = StandardScaler().fit(X_train1)
scaler.transform(X_train1)
scaler.transform(X_test1)
scaler = Normalizer().fit(X_train1)
scaler.transform(X_train1)
scaler.transform(X_test1)

scaler = StandardScaler().fit(X_train2)
scaler.transform(X_train2)
scaler.transform(X_test2)
scaler = Normalizer().fit(X_train2)
scaler.transform(X_train2)
scaler.transform(X_test2)

scaler = StandardScaler().fit(X_train3)
scaler.transform(X_train3)
scaler.transform(X_test3)
scaler = Normalizer().fit(X_train3)
scaler.transform(X_train3)
scaler.transform(X_test3)

#Обучение на очищенных данных

tree_model = tree.DecisionTreeClassifier()
tree_model.fit (X_train1, Y_train1)
predictions1 = tree_model.predict(X_test1)
print("Accuracy: {}".format((tree_model.score(X_test1,Y_test1))*100))

svm_model = svm.SVC(kernel='linear')
svm_model.fit(X_train2, Y_train2)
predictions2 = svm_model.predict(X_test2)
print("Accuracy: {}".format((svm_model.score(X_test2,Y_test2))*100))

lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train3, Y_train3)
predictions3= lda_model.predict(X_test3)
print("Accuracy: {}".format((lda_model.score(X_test3,Y_test3))*100))

#Визуализация

fig = plt.figure(figsize=(100,100))
_ = tree.plot_tree(tree_model, feature_names = ['date', 'time', 'test', 'negative', 'confirmed',
      'released', 'deceased'], class_names = ['0', '1'], filled=True)

colors = {'0': 'red', '1': 'blue'}
graph(X_test2["test"], X_test2["negative"], Y_test2, colors, False)
graph(X_test2["test"], X_test2["negative"], predictions2, colors, True)


colors = {'0': 'red', '1': 'blue'}
graph(X_test3["test"], X_test3["confirmed"], Y_test3, colors, False)
graph(X_test3["test"], X_test3["confirmed"], predictions3, colors, False)
