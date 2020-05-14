import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt
import logisticRegression

# Load Mnist Data set
(x_train, y_train), (x_test, y_test) = load_mnist(flatten=True, normalize=True, one_hot_label=True)
num = np.unique(y_train, axis=0)  # num = y array 중 unique한 값들로만 이루어진 array
num = num.shape[0]  # num = y array 중 unique한 값들의 개수

LRmodel, cost_arr = [], []
i = 0
while i < num:
    print("\n***", i, "th Logistic Regression model ***")
    LRmodel.append(logisticRegression.logisticRegression(x_train, y_train[:, i], 'single'))
    cost_arr.append(LRmodel[i].learn(learning_rate=0.1, epoch=2))
    i += 1

for graph in cost_arr:
    plt.plot(graph[0], graph[1])
plt.title('Binary-Class Model''s Loss Graph')
plt.xlabel('number of iteration')
plt.ylabel('cost')
plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
plt.tight_layout()
plt.show()

# train Data set
i = 0
while i < num:
    print("\n***", i, "th Logistic Regression model ***")
    accuracy = LRmodel[i].predict(x_test, y_test[:, i])
    print("accuracy:", accuracy, "  score:", int(accuracy*y_test.shape[0]), "/", y_test.shape[0])
    i += 1
