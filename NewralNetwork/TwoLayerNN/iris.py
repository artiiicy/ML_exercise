import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import time
import NeuralNetworkclass as NNclass

def one_hot_encoding(input):
    class_num = np.unique(input, axis=0)        # num = y array 중 unique한 값들로만 이루어진 array
    class_num = class_num.shape[0]              # num = y array 중 unique한 값들의 개수
    return np.eye(class_num)[input], class_num  # np.eye = 단위행렬을 만드는 함수. 즉, y의 unique한 개수만큼의 row를 가지는 단위행렬을 만들고 y에 해당하는 row를 추출한다.

# hyper-parameter values
batch_size = 35
epoch_num = 10
learning_rate = 0.0001
hidden_size = 10

# Load Iris Data set
iris = load_iris()

# Parsing the data sets
X = iris.data   # iris data input
y = iris.target # iris target = label : 0, 1, 2
y_name = iris.target_names  # iris target name : Setosa, Versicolor, Virginica

# sKlearn의 데이터분할 내장함수 사용하여 데이터 분할 (train : test = 8 : 2 비율로 분할)
# params : (test_size = 전체 데이터의 몇 %를 test data로 사용할지 지정 / shuffle = 셔플 여부 설정, random_state = 셔플을 위한 시드 값 지정)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2/10, shuffle=True, random_state=int(time.time()))

# One-hot-Encode the y Data set
y_train, class_num = one_hot_encoding(y_train)
# y_test, class_num = one_hot_encoding(y_test)

twoLayerNetwork = NNclass.TwoLayerNetwork(input_size=X.shape[1], hidden_size=hidden_size, output_size=class_num)

# mini-batch 적용
train_size = X_train.shape[0]
batch_size = min(train_size, batch_size)        # out range of index error 방지하기 위함
iter_num = int(train_size / batch_size) + 1     # mini-batch를 통해 train_set의 전체를 학습시키기 위한 iteration 횟수

for i in range(iter_num):
    # 0부터 train data까지의 idx를 각 각 batch_size의 크기를 갖는 batch data로 지정
    from_idx = i * batch_size
    to_idx = min((i * batch_size) + batch_size, train_size) # out range of index error 방지하기 위함

    x_batch = X_train[from_idx:to_idx]
    y_batch = y_train[from_idx:to_idx]

