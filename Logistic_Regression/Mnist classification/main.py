import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from sklearn.model_selection import train_test_split
import time
import random
import matplotlib.pyplot as plt
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
(x_train, y_train), (x_test, y_test) = load_mnist(flatten=True, normalize=True)

# Test를 위한 Data 10000개 중 100개 임의 추출
rand_array = generate_rand_data(len(x_test), 100)
x_test_rand = x_test[rand_array]
y_test_rand = y_test[rand_array]

num = np.unique(y_train, axis=0)  # num = y array 중 unique한 값들로만 이루어진 array
num = num.shape[0]  # num = y array 중 unique한 값들의 개수
y_train = np.eye(num)[y_train]  # np.eye = 단위행렬을 만드는 함수. 즉, y의 unique한 개수만큼의 row를 가지는 단위행렬을 만들고 y에 해당하는 row를 추출한다.

LRmodel = logisticRegression.logisticRegression(0.001, x_train, y_train)

start = time.time()
iteration_array = []
cost_array = []

for i in range(1,501):
    iteration_array.append(i)
    cost_array.append(LRmodel.cost())
    print("epoch:", i, "  cost:", cost_array[i - 1])
    LRmodel.learn()

prediction_Time = time.time() - start
accuracy = LRmodel.predict(x_test, y_test)
print("accuracy:", accuracy, "  Prediction time:", prediction_Time)

plt.plot(iteration_array, cost_array)
plt.title('Loss Graph')
plt.xlabel('number of iteration')
plt.ylabel('cost')
plt.legend(['0','1','2','3','4','5','6','7','8','9'],loc='upper right')
plt.show()
