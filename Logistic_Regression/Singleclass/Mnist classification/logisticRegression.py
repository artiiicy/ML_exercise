import numpy as np
import matplotlib.pyplot as plt

class logisticRegression:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.w = np.random.randn(X.shape[1], y.shape[1])    # numpy의 내장함수를 통해 초기 feature 값 랜덤생성
                                                            # randn : 평균 0, 표준편차 1의 표준정규분포 난수 값 생성
    def show_weight(self):
        print(self.w)

    def show_lossGraph(self, iteration_array, cost_array):
        plt.plot(iteration_array, cost_array)
        plt.title('Loss Graph')
        plt.xlabel('number of iteration')
        plt.ylabel('cost')
        plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
        plt.show()

    def sigmoid(self, X, w):
        z = np.dot(X, w)
        eMin = -np.log(np.finfo(type(0.1)).max)
        zSafe = np.array(np.maximum(z, eMin))
        return (1.0/(1 + np.exp(-zSafe)))

    def hypothesis(self, X, w):
        return 1 / (1 + np.exp(-np.dot(X, w)))

    def cost(self):
        return np.mean(-(self.y * np.log(self.sigmoid(self.X, self.w) + np.finfo(float).eps) + (1 - self.y) * np.log(1 - self.sigmoid(self.X, self.w) + np.finfo(float).eps)), axis=0)

    def learn(self, learning_rate, epoch):
        iteration_array = []
        cost_array = []

        i = 1
        while i <= epoch:
            self.w -= learning_rate * np.dot(self.X.transpose(), (self.sigmoid(self.X, self.w) - self.y))
            iteration_array.append(i)
            cost_array.append(self.cost())
            print("epoch:", i, "  cost:", cost_array[i - 1])
            i += 1
        self.show_lossGraph(iteration_array, cost_array)

    def predict(self, X_test, y_test):
        return 1 - (np.count_nonzero(np.argmax(self.sigmoid(X_test, self.w), axis=1) - y_test) / y_test.shape[0])
