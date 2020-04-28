import sys, os
sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
from PIL import Image
import random
import time

import KNNclass

def img_show(img):
    img = img.reshape(28, 28)
    pil_img = Image.fromarray(np.uint8(img))    # convert numpy array to image
    pil_img.show()

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

# Label name 및 K 값 list에 저장
label_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
k_list = [3]

print("\n*** input feature dimension :", x_test_rand.shape[1], "***")
for k_value in k_list:
    print("k :", k_value)

    new_knn = KNNclass.knn(k=k_value, x=x_train, y=y_train)
    i = 0
    score = 0

    start = time.time() # prediction time 기록을 위하여 학습 시작 시간 기록.
    while i < len(x_test_rand):
        distances = new_knn.cal_distance(x_test_rand[i], y_test_rand[i])
        class_list, dis_list = new_knn.obtain_KNN(distances)
        result = new_knn.obtain_wmv(class_list, dis_list, len(label_name))
        print(i, "th data    Result:", label_name[result], "    Label:", label_name[y_test_rand[i]], "    ", label_name[result]==label_name[y_test_rand[i]])
        i += 1
        score += new_knn.get_score(label_name[result], label_name[y_test_rand[i-1]])

    print('prediction time :', time.time() - start)
    print('accuracy :', '%.2f' % (score/i))
    print('score :', score, '/', i)

print("*** input feature dimension :", int(x_test_rand.shape[1] / 4), "***")
for k_value in k_list:
    print("k :", k_value)

    x_train = x_train.reshape(60000, 28, 28)
    x_train = x_train[:, :, 14:21]
    x_train = x_train.reshape(60000, 28*7)

    x_test_rand = x_test_rand.reshape(100, 28, 28)
    x_test_rand = x_test_rand[:, :, 14:21]
    x_test_rand = x_test_rand.reshape(100, 28*7)

    new_knn = KNNclass.knn(k=k_value, x=x_train, y=y_train)
    i = 0
    score = 0

    start = time.time()
    while i < len(x_test_rand):
        distances = new_knn.cal_distance(x_test_rand[i], y_test_rand[i])
        class_list, dis_list = new_knn.obtain_KNN(distances)
        result = new_knn.obtain_wmv(class_list, dis_list, len(label_name))
        #print(i, "th data    Result:", label_name[result], "    Label:", label_name[y_test_rand[i]], "    ", label_name[result]==label_name[y_test_rand[i]])
        i += 1
        score += new_knn.get_score(label_name[result], label_name[y_test_rand[i-1]])

    print('prediction time :', time.time() - start)
    print('accuracy :', '%.2f' % (score/i))
    print('score :', score, '/', i)
