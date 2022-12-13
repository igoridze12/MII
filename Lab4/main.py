import csv
import sklearn
import pandas
import numpy
import pylab
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def distance(xt1, xt2, xi1, xi2):
    return (abs(xt1 - xi1) ** 2 + abs(xt2 - xi2) ** 2) ** 0.5

#header (row 0)
data =[['Food', 'Sweetness', 'Crustiness', 'Type'],
       #training set of 10 (row 1-10)
       ['Apple', '7', '7', '0'],
       ['Salad', '2', '5', '1'],
       ['Bacon', '1', '2', '2'],
       ['Nuts', '1', '5', '2'],
       ['Fish', '1', '1', '2'],
       ['Cheese', '1', '1', '2'],
       ['Banana', '9', '1', '0'],
       ['Carrot', '2', '8', '1'],
       ['Grape', '8', '1', '0'],
       ['Orange', '6', '1', '0'],
       #test set of 10 (row 11-20)
       ['Strawberry', '9', '1', '0'],
       ['Lettuce', '3', '7', '1'],
       ['Shashlik', '1', '1', '2'],
       ['Pear', '5', '3', '0'],
       ['Celery', '1', '5', '1'],
       ['Apple pie', '6', '10', '0'],
       ['Brownie', '10', '9', '0'],
       ['Puff with cottage cheese', '8', '6', '0'],
       ['Cabbage', '3', '4', '1'],
       ['Cinnamon', '10', '7', '0']]

with open('sw_data_new.csv', 'w', encoding='utf8') as f:
    writer = csv.writer(f)
    for row in data:
        writer.writerow(row)

def knn_manual(data, training_size, types_count, k_max, parzen_window):
    test_size = len(data) - training_size - 1
    new_dist = numpy.zeros((test_size, training_size))

    for i in range(test_size):
        for j in range (training_size):
            new_dist[i][j] = distance(int(data[training_size + i + 1][1]), int(data[training_size + i + 1][2]), int(data[j + 1][1]), int(data[j + 1][2])) #row 0 is header hence +1

    err_k = [0]*k_max
    for k in range(k_max): #k - number of neighbors
        print('\nClassification for k = ', k + 1)
        types = [0] * test_size #estimated types
        err = [0] * test_size #errors

        for i in range(test_size):
            quant_dist = [0] * types_count #how many times each of three types is seen among a point's neighbors
            print('\n\t' + str(i) + '. Classification of ', data[training_size + i][0])
            distances = numpy.array(new_dist[i, :]) #distances of a test point to other training points
            distance_max = max(distances)

            for j in range(k + 1):
                in_min = list(distances).index(min(distances)) #index of element with minimal distance
                quant_dist[int(data[in_min + i + 1][3])] += 1
                if (distances[j] < parzen_window): #parzen window
                    quant_dist[int(data[in_min + 1][3])] += distance_max + distances[j]
                else:
                    quant_dist[int(data[in_min + 1][3])] += distance_max
                distances[in_min] = 1000 #make sure current closest neighbor isn't processed twice
                max1 = max(quant_dist) #how frequently most seen type is seen
                print('\t\tNeighbor index = ' + str(in_min) + ', neighbor - ' + data[in_min + 1][0])
                print('\t\tDistances: ' + str(quant_dist))

            max2 = list(quant_dist).index(max1) #most frequently seen type aka estimated type
            types[i] = max2
            print('\tEstimated type: ', types[i])
            print('\tActual type: ', int(data[training_size + i + 1][3]))
            if (int(types[i]) == int(data[training_size + i + 1][3])): #if estimated and actual types match
                print('\tMatch')
                err[i] = 0 #no error
            else: #if estimated and actual types don't match
                print('\tNo match')
                err[i] = 1 #error

        err_k[k] = numpy.mean(err) #mean error for each neighbor
        print('Errors: ', err_k)

    return types, err_k

def knn_sklearn(values, training_size, types, k_max):
    X_train, X_test, y_train, y_test = train_test_split(values, types, test_size = (int(values.size/2 - training_size)), random_state = 0)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = KNeighborsClassifier(n_neighbors = k_max)
    model.fit(X_train, y_train)

    estimations = model.predict(X_test)

    print('\nTraining set parameters:')
    print(X_train)
    print('\nTest set parameters:')
    print(X_test)
    print('\nTraining set types:')
    print(y_train)
    print('\nTest set types:')
    print(y_test)
    print('\nEstimations:')
    print(estimations)

    return X_train, X_test, y_train, y_test, estimations

def charts(k_max, err_k, sweetness, crustiness, colors, types_original, types_estimated):
    pylab.subplot(3, 1, 1)
    pyplot.plot([i for i in range(1, k_max + 1)], err_k)
    pyplot.title('Dependence of error on k')
    pyplot.xlabel('k')
    pyplot.ylabel('Error')

    color_list = [colors[str(i)] for i in types_original]
    pylab.subplot(3, 1, 2)
    pyplot.scatter(sweetness, crustiness, c=color_list)
    pyplot.title('Original data')
    pyplot.xlabel('Sweetness')
    pyplot.ylabel('Crustiness')

    color_list = [colors[str(i)] for i in types_estimated]
    pylab.subplot(3, 1, 3)
    pyplot.scatter(sweetness, crustiness, c=color_list)
    pyplot.title('Estimated data')
    pyplot.xlabel('Sweetness')
    pyplot.ylabel('Crustiness')
    pyplot.show()

training_size = 10
k_max = 5
parzen_window = 2
types_count = 3

estimations, err_k = knn_manual(data, training_size, types_count, k_max, parzen_window)
dataset = pandas.read_csv('sw_data_new.csv')
sweetness = dataset['Sweetness']
crustiness = dataset['Crustiness']
colors = {'0': 'black', '1': 'green', '2': 'red', '3': 'blue'}
types_original = dataset['Type']
types_training = dataset[:training_size]['Type']
types_estimated_series = pandas.Series(estimations)
types_final = pandas.concat([types_training, types_estimated_series])
charts(k_max, err_k, sweetness, crustiness, colors, types_original, types_final)

values = numpy.array(list(zip(sweetness, crustiness)), dtype=numpy.float64)
X_train, X_test, y_train, y_test, estimations = knn_sklearn(values, training_size, types_original, k_max)
estimations_series = pandas.Series(estimations)
types_training_series = pandas.Series(types_training)
types_estimated = pandas.concat([types_training_series, estimations_series])
err = 0
counter = 0
actual_types = pandas.Series(dataset['Type'])
types_final = pandas.concat([pandas.Series(dataset[:training_size]['Type']), estimations_series])
print('\nCalculating errors for scikit-learn\'s knn')
for i in types_final:
    print('Estimated type: ' + str(i))
    print('Actual type: ' + str(actual_types[counter]))
    if (i == actual_types[counter]):
        err += 0
    else:
        err += 1
    counter += 1
    err = err / counter
    print('Error: ' + str(err) + '\n')
    err_k = []
    for i in range(1, k_max + 1):
        err_k.append(err)
charts(k_max, err_k, sweetness, crustiness, colors, types_original, types_estimated)

#repeat the experiment with an additional data type

#header (row 0)
data =[['Food', 'Sweetness', 'Crustiness', 'Type'],
       #training set of 14 (row 1-14)
       ['Apple', '7', '7', '0'],
       ['Salad', '2', '5', '1'],
       ['Bacon', '1', '2', '2'],
       ['Nuts', '1', '5', '2'],
       ['Fish', '1', '1', '2'],
       ['Cheese', '1', '1', '2'],
       ['Banana', '9', '1', '0'],
       ['Carrot', '2', '8', '1'],
       ['Grape', '8', '1', '0'],
       ['Orange', '6', '1', '0'],
       ['Corn sticks', '4', '5', '3'],
       ['Melon', '5', '7', '3'],
       ['Potato chips', '4', '10', '3'],
       ['Corn chips', '5', '8', '3'],
       #test set of 12 (row 15-26)
       ['Strawberry', '9', '1', '0'],
       ['Lettuce', '3', '7', '1'],
       ['Shashlik', '1', '1', '2'],
       ['Pear', '5', '3', '0'],
       ['Celery', '1', '5', '1'],
       ['Apple pie', '6', '10', '0'],
       ['Brownie', '10', '9', '0'],
       ['Puff with cottage cheese', '8', '6', '0'],
       ['Cabbage', '3', '4', '1'],
       ['Cinnamon', '10', '7', '0'],
       ['Muesli', '4', '8', '3'],
       ['Nori chips', '5', '9', '3']]

with open('sw_data_new.csv', 'w', encoding='utf8') as f:
    writer = csv.writer(f)
    for row in data:
        writer.writerow(row)

training_size = 14
k_max = 5
parzen_window = 2
types_count = 4

estimations, err_k = knn_manual(data, training_size, types_count, k_max, parzen_window)
dataset = pandas.read_csv('sw_data_new.csv')
sweetness = dataset['Sweetness']
crustiness = dataset['Crustiness']
colors = {'0': 'black', '1': 'green', '2': 'red', '3': 'blue'}
types_original = dataset['Type']
types_training = dataset[:training_size]['Type']
types_estimated_series = pandas.Series(estimations)
types_final = pandas.concat([types_training, types_estimated_series])
charts(k_max, err_k, sweetness, crustiness, colors, types_original, types_final)

values = numpy.array(list(zip(sweetness, crustiness)), dtype=numpy.float64)
X_train, X_test, y_train, y_test, estimations = knn_sklearn(values, training_size, types_original, k_max)
estimations_series = pandas.Series(estimations)
types_training_series = pandas.Series(types_training)
types_estimated = pandas.concat([types_training_series, estimations_series])
err = 0
counter = 0
actual_types = pandas.Series(dataset['Type'])
types_final = pandas.concat([pandas.Series(dataset[:training_size]['Type']), estimations_series])
print('\nCalculating errors for scikit-learn\'s knn')
for i in types_final:
    print('Estimated type: ' + str(i))
    print('Actual type: ' + str(actual_types[counter]))
    if (i == actual_types[counter]):
        err += 0
    else:
        err += 1
    counter += 1
    err = err / counter
    print('Error: ' + str(err) + '\n')
    err_k = []
    for i in range(1, k_max + 1):
        err_k.append(err)
charts(k_max, err_k, sweetness, crustiness, colors, types_original, types_estimated)
