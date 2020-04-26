import sys, os
sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

import KNNclass

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))    # convert numpy array to image
    pil_img.show()

# Load Mnist Data set
(x_train, y_train), (x_test, y_test) = load_mnist(flatten=True, normalize=False)

# 컴퓨터 과부화를 줄이기 위하여 Train Data의 일부만을 사용.
x_train = x_train[:10]
y_train = y_train[:10]

image = x_train[0]
label = y_train[0]

image = image.reshape(28,28)
img_show(image)

