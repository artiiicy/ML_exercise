import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from scipy.misc import derivative
import time
import matplotlib.pyplot as plt

class TwoLayerNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # set parameters with random value
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size)    # 1번째 층의 가중치
        self.params['b1'] = np.random.randn(hidden_size)                # 1번째 층의 편향
        self.params['W2'] = np.random.randn(hidden_size, output_size)   # 2번째 층의 가중치
        self.params['b2'] = np.random.randn(output_size)                # 2번째 층의 편향

        # Load Iris Data set
        iris = load_iris()

        # Parsing the data sets
        X = iris.data  # iris data input
        t = iris.target  # iris target = label : 0, 1, 2

        # sKlearn의 데이터분할 내장함수 사용하여 데이터 분할 (train : test = 8 : 2 비율로 분할)
        # params : (test_size = 전체 데이터의 몇 %를 test data로 사용할지 지정 / shuffle = 셔플 여부 설정, random_state = 셔플을 위한 시드 값 지정)
        self.X_train, self.X_test, self.t_train, self.t_test = train_test_split(X, t, test_size=2 / 10, shuffle=True, random_state=int(time.time()))

        # One-hot-Encode the y Data set
        self.t_train = self.one_hot_encoding(self.t_train)
        # self.t_test, class_num = one_hot_encoding(self.t_test)

    def one_hot_encoding(self, input):
        class_num = np.unique(input, axis=0)            # num = y array 중 unique한 값들로만 이루어진 array
        self.class_num = class_num.shape[0]             # num = y array 중 unique한 값들의 개수

        return np.eye(self.class_num)[input]            # np.eye = 단위행렬을 만드는 함수. 즉, y의 unique한 개수만큼의 row를 가지는 단위행렬을 만들고 y에 해당하는 row를 추출한다.

    # Loss Function - Cross Entropy Error
    def cross_entropy_error(self, y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        batch_size = y.shape[0]
        return -np.sum(t * np.log(y + np.finfo(float).eps)) / batch_size # overflow error 방지를 위하여 매우 작은 수(epsilon)을 더해준다.

    # Activation Function - Sigmoid
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def softmax(self, x):
        exp_a = np.exp(x.T - np.max(x, axis=1))
        exp_a = exp_a.T
        sum_exp_a = np.sum(exp_a, axis=1)
        return np.transpose(exp_a.T / sum_exp_a)

    # 수치미분 함수
    def numerical_gradient_function(self, f, x):
        h = 1e-4
        grad = np.zeros_like(x) # x와 같은 shape의 0배열 생성

        # bias 미분
        if x.ndim == 1:
            for i in range(x.shape[0]):
                xi = x[i]
                x[i] = xi + h
                fx1 = f(x[i])
                x[i] = xi - h
                fx2 = f(x[i])
                grad[i] = (fx1 - fx2) / (2 * h)
                x[i] = xi
                # grad = derivative(f, x[i], h)
                # print(derivative(f, x[i], h))

        # weight 미분
        else:
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    xi = x[i][j]
                    x[i][j] = xi + h
                    fx1 = f(x[i][j])
                    x[i][j] = xi - h
                    fx2 = f(x[i][j])
                    grad[i][j] = (fx1 - fx2) / (2 * h)
                    x[i][j] = xi
                    # grad[i] = derivative(f, x[i][j], h)
                    # print(derivative(f, x[i][j], h))

        return grad

    # input(x)에 대하여 결과 추정하여 반환하는 함수
    def predict(self, x):
        z1 = np.dot(x, self.params['W1']) + self.params['b1']
        a1 = self.sigmoid(z1)
        z2 = np.dot(a1, self.params['W2']) + self.params['b2']
        y = self.softmax(z2)

        return y

    # input(x)에 대한 추정값과 실제 값(t)의 차이를 구하는 함수
    def loss(self, x, t):
        # predict 함수 호출하여 input(x)에 대한 추정값 도출
        y = self.predict(x)

        # Cross Entropy Error 함수 통하여 추정값과 실제값 차이 도출하여 반환
        cost = self.cross_entropy_error(y, t)
        return cost

    def accuracy(self, x, t):
        # 내 코드 아니아ㅓ니아ㅓ린어랸야야야어야ㅓ이ㅏ너리ㅏㄴ어리ㅏㄴ어리ㅏㄴ어리ㅏ너
        y = self.predict(x)

        y = np.argmax(y, axis=1)
        # print(y)
        t = np.argmax(t, axis=1)
        # print(t)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        # print((1 - np.count_nonzero(np.argmax(self.predict(x), axis=1) - np.argmax(t)) / t.shape[0]))

    # 추정값과 실제값의 차이(cost)에서의 Loss function 미분값 반환하는 함수
    def numerical_gradient(self, x, t):
        # 함수를 매개변수로 넘겨주기 위하여 python의 lambda기법 사용
        f = lambda w: self.loss(x, t)

        grad = {}
        grad['W1'] = self.numerical_gradient_function(f, self.params['W1'])
        grad['W2'] = self.numerical_gradient_function(f, self.params['W2'])
        grad['b1'] = self.numerical_gradient_function(f, self.params['b1'])
        grad['b2'] = self.numerical_gradient_function(f, self.params['b2'])

        # print(grad)
        return grad

    def learn(self, lr, epoch, batch_size):
        # mini-batch 적용
        train_size = self.X_train.shape[0]
        batch_size = min(train_size, batch_size)  # out range of index error 방지하기 위함
        iter_num = int(train_size / batch_size) + 1  # mini-batch를 통해 train_set의 전체를 학습시키기 위한 iteration 횟수

        cost_array, accuracy_array = [], []
        for k in range(epoch):
            shuffle_idx = np.arange(self.X_train.shape[0])
            np.random.shuffle(shuffle_idx)
            self.X_train = self.X_train[shuffle_idx]
            self.t_train = self.t_train[shuffle_idx]
            cost = 0

            # mini-batch
            for i in range(iter_num):
                # 0부터 train data까지의 idx를 각 각 batch_size의 크기를 갖는 batch data로 지정
                from_idx = i * batch_size
                to_idx = min((i * batch_size) + batch_size, train_size)  # out range of index error 방지하기 위함

                if from_idx >= train_size:
                    break

                x_batch = self.X_train[from_idx:to_idx]
                t_batch = self.t_train[from_idx:to_idx]

                # 편미분 방정식을 이용하여 기울기 값 도출
                grad = self.numerical_gradient(x_batch, t_batch)

                # weight값을 기울기 방향으로 조정하여 학습
                for weight in self.params:
                    self.params[weight] -= lr * grad[weight]

            # epoch에서의 cost, accuracy값 도출
            cost_array.append(self.loss(self.X_train, self.t_train))
            accuracy_array.append(self.accuracy(self.X_train, self.t_train))
            print(cost_array[k])

        plt.plot(list(range(epoch)), cost_array)
        plt.plot(list(range(epoch)), accuracy_array)
        plt.title('Training Loss & Accuracy Graph')
        plt.xlabel('number of iteration')
        plt.ylabel('cost & accuracy')
        plt.legend(['loss', 'accuracy'], loc='upper right')
        plt.show()
