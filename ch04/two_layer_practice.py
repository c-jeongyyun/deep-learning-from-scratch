import sys, os

import numpy as np

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록
from common.functions import *
from common.gradient import numerical_gradient


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)  # 가중치 초기화

        self.params["b1"] = np.zeros(hidden_size)  # 편향 초기화
        self.params["W1"] = weight_init_std * np.random.randn(hidden_size, output_size)  # 가중치 초기화
        self.params["b2"] = np.zeros(output_size)  # 편향 초기화

    def predict(self, x):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape(0))
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W : self.loss(x,t)

        grad = {"W1": numerical_gradient(loss_W, self.params["W1"]),
                "b1": numerical_gradient(loss_W, self.params["b1"]),
                "W2": numerical_gradient(loss_W, self.params["W2"]),
                "b2": numerical_gradient(loss_W, self.params["b2"])}

        return grad 
