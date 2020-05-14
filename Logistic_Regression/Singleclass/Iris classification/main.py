import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import logisticRegression

def one_hot_encoding(input):
    num = np.unique(input, axis=0)  # num = y array 중 unique한 값들로만 이루어진 array
    num = num.shape[0]  # num = y array 중 unique한 값들의 개수
    return np.eye(num)[input], num  # np.eye = 단위행렬을 만드는 함수. 즉, y의 unique한 개수만큼의 row를 가지는 단위행렬을 만들고 y에 해당하는 row를 추출한다.

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
y_train, num = one_hot_encoding(y_train)
y_test, num = one_hot_encoding(y_test)

LRmodel, cost_arr = [], []
i = 0
while i < num:
    print("\n***", i, "th Logistic Regression model ***")
    LRmodel.append(logisticRegression.logisticRegression(X_train, y_train[:, i], 'single'))
    cost_arr.append(LRmodel[i].learn(learning_rate=0.0001, epoch=100))
    i += 1

i = 1
for graph in cost_arr:
    plt.subplot(num, 2, i)
    plt.plot(graph[0], graph[1])
    plt.title('%s-th Model''s Loss Graph' %str(i-1))
    plt.xlabel('number of iteration')
    plt.ylabel('cost')
    plt.legend(str(i-1), loc='upper right')
    i += 1
plt.tight_layout()
plt.show()

# train Data set
i = 0
while i < num:
    print("\n***", i, "th Logistic Regression model ***")
    accuracy = LRmodel[i].predict(X_test, y_test[:, i])
    print("accuracy:", accuracy, "  score:", int(accuracy*y_test.shape[0]), "/", y_test.shape[0])
    i += 1
