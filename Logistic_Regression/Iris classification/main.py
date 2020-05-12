import sys, os
sys.path.append(os.pardir)
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import logisticRegression

# Load Iris Data set
iris = load_iris()

# Parsing the data sets
X = iris.data   # iris data input
y = iris.target # iris target = label : 0, 1, 2
y_name = iris.target_names  # iris target name : Setosa, Versicolor, Virginica

# Divide data sets into train, test sets
X_train, X_test, y_train, y_test\
    = train_test_split(X, y, test_size=1/15, shuffle=True, random_state=int(time.time()))  # sklearn의 데이터분할 내장함수 사용.
                                                                                           # test_size : 전체 데이터의 몇 %를 test data로 사용할지 지정
                                                                                           # shuffle : 셔플 여부 설정, random_state : 셔플을 위한 시드 값 지정

num = np.unique(y_train, axis=0)  # num = y array 중 unique한 값들로만 이루어진 array
num = num.shape[0]  # num = y array 중 unique한 값들의 개수
y_train = np.eye(num)[y_train]  # np.eye = 단위행렬을 만드는 함수. 즉, y의 unique한 개수만큼의 row를 가지는 단위행렬을 만들고 y에 해당하는 row를 추출한다.

LRmodel = logisticRegression.logisticRegression(0.001, X_train, y_train)

start = time.time()
iteration_array = []
cost_array = []

for i in range(1,101):
    iteration_array.append(i)
    cost_array.append(LRmodel.cost())
    print("epoch:", i, "  cost:", cost_array[i - 1])
    LRmodel.learn()

prediction_Time = time.time() - start
accuracy = LRmodel.predict(X_test, y_test)
print("accuracy:", accuracy, "  Prediction time:", prediction_Time)

plt.plot(iteration_array, cost_array)
plt.xlabel('number of iteration')
plt.ylabel('cost')
plt.show()
