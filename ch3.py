import sys, os
import numpy as np
import pickle
# sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

from PIL import Image


def img_show(img):
    # img should be 28x28 integer
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


img = x_train[0]
img=img.reshape(28,28)
img_show(img)

def init_network():
    with open('sample_weight.pkl', 'rb') as f:
        network=pickle.load(f)
    return network

network=init_network()
print(network['W1'].shape)


def predict(network, x):
    W1, W2, W3=network['W1'], network['W2'], network['W3']
    b1, b2, b3=network['b1'],network['b2'],network['b3']

    a1=np.dot(x,W1)+b1
    z1=sigmoid(a1)
    a2=np.dot(z1, W2)+b2
    z2=sigmoid(a2)
    a3=np.dot(a2, W3)+b3
    y=softmax(a3)

    return y



