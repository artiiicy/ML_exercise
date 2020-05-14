import numpy as np
import matplotlib.pyplot as plt

class logisticRegression:
    def __init__(self, X, y, str):
        self.X = X
        self.y = y

        if str == 'multi':
            self.type = 'm'
        elif str == 'single':
            self.type = 's'

        if y.ndim == 1:
            self.w = np.random.randn(X.shape[1])
        else:
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
        iteration_arr = []
        cost_arr = []

        i = 1
        while i <= epoch:
            self.w -= learning_rate * np.dot(self.X.transpose(), (self.sigmoid(self.X, self.w) - self.y))
            iteration_arr.append(i)
            cost_arr.append(self.cost())
            print("epoch:", i, "  cost:", cost_arr[i - 1])
            i += 1
        
        if self.type == 'm':
            self.show_lossGraph(iteration_arr, cost_arr)
        return iteration_arr, cost_arr

    def predict(self, X_test, y_test):
        if self.type == 'm':
            return 1 - (np.count_nonzero(np.argmax(self.sigmoid(X_test, self.w), axis=1) - y_test) / y_test.shape[0])
        else:
            return np.sum((self.sigmoid(X_test, self.w) > 0.5) == y_test) / y_test.shape[0]
