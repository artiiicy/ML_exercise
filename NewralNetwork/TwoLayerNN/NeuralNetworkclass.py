import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt

class TwoLayerNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size    # input layer의 노드 개수
        self.hidden_size = hidden_size  # hidden layer의 노드 개수
        self.output_size = output_size  # output layer의 노드 개수

        # set parameters with random value
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size)    # 1번째 층의 가중치
        self.params['b1'] = np.random.randn(hidden_size)                # 1번째 층의 편향
        self.params['W2'] = np.random.randn(hidden_size, output_size)   # 2번째 층의 가중치
        self.params['b2'] = np.random.randn(output_size)                # 2번째 층의 편향

        # Load Iris Data set
        iris = load_iris()

        # Parsing the data sets
        X = iris.data       # iris input data
        t = iris.target     # iris target

        # train, test data set으로 데이터 분할
        # sKlearn의 데이터분할 내장함수 사용하여 데이터 분할 (train : test = 8 : 2 비율로 분할)
        # params : (test_size = 전체 데이터의 몇 %를 test data로 사용할지 지정 / shuffle = 셔플 여부 설정, random_state = 셔플을 위한 시드 값 지정)
        self.X_train, self.X_test, self.t_train, self.t_test = train_test_split(X, t, test_size=2 / 10, shuffle=True, random_state=int(time.time()))

        # One-hot-Encode the y Data set
        self.t_train = self.one_hot_encoding(self.t_train)
        self.t_test = self.one_hot_encoding(self.t_test)

    # One Hot Encoding Function
    def one_hot_encoding(self, input):
        class_num = np.unique(input, axis=0)            # input array 중 unique한 값들로만 이루어진 array 생성
        self.class_num = class_num.shape[0]             # input array 중 unique한 값들의 개수

        return np.eye(self.class_num)[input]            # np.eye = 단위행렬을 만드는 함수. 즉, class_num만큼의 row를 가지는 단위행렬을 만들고 input array에 해당하는 row를 반환

    # Loss Function - Cross Entropy Error
    def cross_entropy_error(self, y, t):
        # mini-batch로 일반화하기 위해 y가 1차원일때는 reshape
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        batch_size = y.shape[0]
        return -np.sum(t * np.log(y + np.finfo(float).eps)) / batch_size    # overflow error 방지를 위하여 매우 작은 수(epsilon)을 더함

    # Activation Function - Sigmoid
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Activation Function - Softmax
    def softmax(self, x):
        expArr = np.exp(x.T - np.max(x, axis=1))     # x array의 행 단위 최댓값 산출하여 정규화
        expArr = expArr.T
        sumArr = np.sum(expArr, axis=1)              # 행의 합이 1이 되도록 정규화
        return np.transpose(expArr.T / sumArr)

    # 수치미분 함수 - 기울기 반환
    def numerical_gradient_function(self, f, x):
        h = 1e-4
        grad = np.zeros_like(x)  # x와 shape이 같은 0 배열 생성

        # bias 편미분 (1차원 배열)
        if x.ndim == 1:
            for i in range(x.shape[0]):
                temp = x[i]
                x[i] = temp + h     # f(x+h)
                fxh1 = f(x)
                x[i] = temp - h     # f(x-h)
                fxh2 = f(x)
                grad[i] = (fxh1 - fxh2) / (2 * h)   # 미분공식 f(x+h) - f(x-h) / 2*h
                x[i] = temp

        # weight 편미분 (2차원 배열)
        else:
            idx = 0
            for i in x:
                tempGrad = np.zeros_like(i)

                for j in range(i.shape[0]):
                    temp = i[j]
                    i[j] = temp + h     # f(x+h)
                    fxh1 = f(x)
                    i[j] = temp - h     # f(x-h)
                    fxh2 = f(x)
                    tempGrad[j] = (fxh1 - fxh2) / (2 * h)   # 미분공식 f(x+h) - f(x-h) / 2*h
                    i[j] = temp
                grad[idx] = tempGrad
                idx += 1

        return grad

    # input(x)에 대하여 결과 추정하여 반환하는 함수
    def predict(self, x):
        z1 = np.dot(x, self.params['W1']) + self.params['b1']
        a1 = self.sigmoid(z1)
        z2 = np.dot(a1, self.params['W2']) + self.params['b2']
        y = self.softmax(z2)

        return y

    # input(x)에 대한 추정값과 실제 값(t)의 차이(=loss)를 구하는 함수
    def loss(self, x, t):
        # predict 함수 호출하여 input(x)에 대한 추정값 도출
        y = self.predict(x)

        # Cross Entropy Error 함수 통하여 추정값과 실제값 차이 도출하여 반환
        cost = self.cross_entropy_error(y, t)
        return cost

    # input(x)에 대한 추정값과 실제 값(t) 비교를 통해 정확도 도출
    def accuracy(self, x, t):
        y = self.predict(x)

        return ((1 - np.count_nonzero(np.argmax(y, axis=1) - np.argmax(t, axis=1)) / t.shape[0]))

    # weight에 대한 Loss function 편미분값(=기울기값) 반환하는 함수
    def numerical_gradient(self, x, t):
        # 함수를 매개변수로 넘겨주기 위하여 python의 lambda기법 사용
        f = lambda g: self.loss(x, t)

        # weight (W1, W2, b1, b2)에 대한 편미분값(=기울기값) 저장 후 반환
        grad = {}
        grad['W1'] = self.numerical_gradient_function(f, self.params['W1'])
        grad['W2'] = self.numerical_gradient_function(f, self.params['W2'])
        grad['b1'] = self.numerical_gradient_function(f, self.params['b1'])
        grad['b2'] = self.numerical_gradient_function(f, self.params['b2'])

        return grad

    # 학습 함수. params : (lr = Learning rate, epoch = 학습 반복 수, batch_size = mini-batch 크기, verbose = cost, accuracy 출력 여부)
    def learn(self, lr, epoch, batch_size, verbose):
        # mini-batch 적용
        train_size = self.X_train.shape[0]
        batch_size = min(train_size, batch_size)        # out range of index error 방지하기 위함
        iter_num = int(train_size / batch_size) + 1     # mini-batch를 통해 train_set의 전체를 학습시키기 위한 iteration 횟수

        cost_array, accuracy_array = [], []
        for k in range(epoch):
            # mini-batch 사용하므로 학습 시 데이터의 순서를 랜덤하게 섞는 코드
            shuffle_idx = np.arange(self.X_train.shape[0])
            np.random.shuffle(shuffle_idx)
            self.X_train = self.X_train[shuffle_idx]
            self.t_train = self.t_train[shuffle_idx]
            cost = 0

            # mini-batch
            for i in range(iter_num):
                # batch_size 크기의 batch data 지정
                from_idx = i * batch_size
                to_idx = min((i * batch_size) + batch_size, train_size)  # out range of index error 방지하기 위함

                # mini-batch의 시작 idx가 train_size 넘어가면 종료
                if from_idx >= train_size:
                    break

                # Set mini-batch data
                x_batch = self.X_train[from_idx:to_idx]
                t_batch = self.t_train[from_idx:to_idx]

                # 편미분 방정식을 이용하여 기울기 값 도출
                grad = self.numerical_gradient(x_batch, t_batch)

                # 각 weight값을 기울기 방향으로 조정하여 학습
                for weight in self.params:
                    self.params[weight] -= lr * grad[weight]

            # epoch에서의 cost, accuracy값 도출
            cost_array.append(self.loss(self.X_train, self.t_train))
            accuracy_array.append(self.accuracy(self.X_train, self.t_train))

            if verbose==True:
                print(k, "- cost, accuracy :", cost_array[k], accuracy_array[k])

        print("Training Accuracy =", np.average(accuracy_array))
        print("Test Accuracy =", self.accuracy(self.X_test, self.t_test))

        # Plot the Loss & Accuracy Graph
        plt.plot(list(range(epoch)), cost_array)
        plt.plot(list(range(epoch)), accuracy_array)
        plt.title('Training Loss & Accuracy Graph')
        plt.xlabel('number of iteration')
        plt.ylabel('cost & accuracy')
        plt.legend(['loss', 'accuracy'], loc='upper right')
        plt.show()
