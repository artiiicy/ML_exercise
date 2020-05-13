import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
import random
import logisticRegression

def generate_rand_data(origin_size, rand_size):
    rand_array = np.zeros(origin_size, dtype=bool)
    ran_num = random.randint(0, origin_size - 1)

    for i in range(rand_size):
        while rand_array[ran_num] == 1:
            ran_num = random.randint(0, origin_size - 1)
        rand_array[ran_num] = 1
    return rand_array

# Load Mnist Data set
(x_train, y_train), (x_test, y_test) = load_mnist(flatten=True, normalize=True)#, one_hot_label=True)
num = np.unique(y_train, axis=0)  # num = y array 중 unique한 값들로만 이루어진 array
num = num.shape[0]  # num = y array 중 unique한 값들의 개수
y_train = np.eye(num)[y_train]  # np.eye = 단위행렬을 만드는 함수. 즉, y의 unique한 개수만큼의 row를 가지는 단위행렬을 만들고 y에 해당하는 row를 추출한다.

LRmodel = logisticRegression.logisticRegression(x_train, y_train)

# train Data set
LRmodel.learn(learning_rate=0.001, epoch=10)
accuracy = LRmodel.predict(x_test, y_test)
print("accuracy:", accuracy, "  score:", int(accuracy*y_test.shape[0]), "/", y_test.shape[0])
