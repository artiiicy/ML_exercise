import numpy as np
from sklearn.datasets import load_iris
import KNNclass

# Load the Iris data sets
iris = load_iris()

# Parsing the data sets
X = iris.data   # iris data input
y = iris.target # iris target = label : 0, 1, 2
y_name = iris.target_names  # iris target name : Setosa, Versicolor, Virginica

k_list = [3, 5, 10]

# Divide data sets into train, test sets
x_train = np.zeros((len(X),)).astype(np.bool)
for i in range(1, len(X)+1):
    if i % 15 == 0:
        x_train[i - 1] = False
    else:
        x_train[i - 1] = True
x_test = ~np.array(x_train)

x_train = np.array(X[x_train])
x_test = np.array(X[x_test])

y_train = np.zeros((len(y),)).astype(np.bool)
for i in range(1, len(y)+1):
    if i % 15 == 0:
        y_train[i - 1] = False
    else:
        y_train[i - 1] = True
y_test = ~np.array(y_train)

y_train = np.array(y[y_train])
y_test = np.array(y[y_test])

print("*** Majority Vote Method ***")
for k_value in k_list:
    print("k :", k_value)
    new_knn = KNNclass.knn(k=k_value, x=x_train, y=y_train)

    i = 0
    while i < len(x_test):
        distances = new_knn.cal_distance(x_test[i], y_test[i])
        class_list, dis_list = new_knn.obtain_KNN(distances)
        result = new_knn.obtain_mv(class_list, len(y_name))
        print("Test Data Index:", i, "Computed class:", y_name[result], "    True class:", y_name[y_test[i]])
        i += 1

print("\n*** Weighted Majority Vote Method ***")
for k_value in k_list:
    print("k :", k_value)
    new_knn = KNNclass.knn(k=k_value, x=x_train, y=y_train)

    i = 0
    while i < len(x_test):
        distances = new_knn.cal_distance(x_test[i], y_test[i])
        class_list, dis_list = new_knn.obtain_KNN(distances)
        result = new_knn.obtain_wmv(class_list, dis_list, len(y_name))
        print("Test Data Index:", i, "Computed class:", y_name[result], "    True class:", y_name[y_test[i]])
        i += 1
