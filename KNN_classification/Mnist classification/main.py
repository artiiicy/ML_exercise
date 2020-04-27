import sys, os
sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

import KNNclass

def img_show(img):
    img = img.reshape(28, 28)
    pil_img = Image.fromarray(np.uint8(img))    # convert numpy array to image
    pil_img.show()

# Load Mnist Data set
(x_train, y_train), (x_test, y_test) = load_mnist(flatten=True, normalize=True)

# 컴퓨터 과부화를 줄이기 위하여 Train Data의 일부만을 사용.
x_train = x_train[:30000]
y_train = y_train[:30000]
x_test = x_test[:100]
y_test = y_test[:100]
label_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
k_list = [3]

for k_value in k_list:
    print("k :", k_value)
    new_knn = KNNclass.knn(k=k_value, x=x_train, y=y_train)

    i = 0
    score = 0
    while i < len(x_test):
        distances = new_knn.cal_distance(x_test[i], y_test[i])
        class_list, dis_list = new_knn.obtain_KNN(distances)
        result = new_knn.obtain_mv(class_list, len(label_name))
        print("Test Data Index:", i, "    Computed class:", label_name[result], "    True class:", label_name[y_test[i]])
        i += 1
        score += new_knn.get_score(label_name[result], label_name[y_test[i-1]])

    print('accuracy :', '%.2f' % (score/i))
