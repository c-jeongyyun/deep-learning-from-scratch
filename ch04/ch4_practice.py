import numpy as np
import sys, os

sys.path.append(os.pardir)
from dataset.mnist import load_mnist


# 오차 제곱합
def sum_square_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


# cross entropy 오차 함수
def cross_entropy_error_one_hot(y, t):  # 추론 결과를 one-hot으로 표현하는 경우
    delta = 1e-7
    if y.ndim == 1:  # 1차원인 경우에만!
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + delta)) / batch_size


def cross_entropy_error_no_one_hot(y, t):  # 추론 결과를 one-hot으로 표현하지 않는    경우
    delta = 1e-7
    if y.ndim == 1:  # 1차원인 경우에만!
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t]) + delta) / batch_size


def numerical_gradient_no_batch(f, x):
    h= 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size) :
        tmp_val = x[idx]
        x[idx] = tmp_val +h
        fxh1 = f(x) # 현재 인덱스를 제외한 나머지는 동일하게 계산해주기 위함

        x[idx]  = tmp_val -h
        fxh2 = f(x)

        grad[idx]  = (fxh1-fxh2) /(2*h)
        x[idx] = tmp_val # 원래 값으로 복구

    return grad

def gradient_descent(f , init_x, lr=0.01, step_num = 100) :
    x = init_x
    for i in range(100) : # 100번 업데이트할거라는 뜻
        grad = numerical_gradient_no_batch(f, x)
        x -= lr*grad  # 기울기 * learning rate 만큼 업데이트 
    return x


if __name__ == "__main__":
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    # 미니배치 - 무작위로 10개의 데이터 가져오기
    train_size = x_train.shape[0]  # 트레이닝 데이터 개수
    batch_size = 10
    batch_mask = np.random.choice(train_size, batch_size)  # random한 인덱스 batch_size개 만큼 반환
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
