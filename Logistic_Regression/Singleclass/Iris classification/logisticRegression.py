import numpy as np

class logisticRegression:
    def __init__(self, learning_rate, X, y):
        self.learning_rate = learning_rate
        self.X = X
        self.y = y
        self.w = np.random.randn(X.shape[1], y.shape[1])    # numpy의 내장함수를 통해 초기 feature 값 랜덤생성
                                                            # randn : 평균 0, 표준편차 1의 표준정규분포 난수 값 생성
    def show_weight(self):
        print(self.w)

    def hypothesis(self, X, w):
        return 1 / (1 + np.exp(-np.dot(X, w)) + np.finfo(float).eps)

    def cost(self):
        return np.mean(-(self.y * np.log(self.hypothesis(self.X, self.w)) + (1 - self.y) * np.log(1 - self.hypothesis(self.X, self.w))), axis=0)

    def learn(self):
        self.w -= self.learning_rate * np.dot(self.X.transpose(), (self.hypothesis(self.X, self.w) - self.y))

    def predict(self, X_test, y_test):
        return 1 - (np.count_nonzero(np.argmax(self.hypothesis(X_test, self.w), axis=1) - y_test) / y_test.shape[0])

