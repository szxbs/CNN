# -*- coding: utf-8 -*-
'''
首先我们确定好有多少类



卷积层
Relu层
池化层
全连接层
softmax
交叉熵损失
'''

import numpy as np
import matplotlib.pyplot as plt
from functions import *
from collections import OrderedDict


# 面向对象仅仅创建这个层数会用到的参数，用完就不使用的则不创建

class Conv():
    def __init__(self, W, B, Stride=1, pad=0):
        self.W = W
        self.B = B
        self.Stride = Stride
        self.pad = pad

        self.X = None
        self.col = None
        self.col_W = None

        self.dW = None
        self.dB = None

    def forward(self, X):
        N, C, K, K = self.W.shape  # 分别是通道数，核数，卷积核宽度，高度
        n, C, L, L = X.shape
        L_out = int((L + self.pad - K) / self.Stride + 1)
        col = im2col(X, K, self.pad, self.Stride)
        col_W = self.W.reshape(N, -1).T

        Z = col @ col_W + self.B
        Z = Z.reshape(n, L_out, L_out, N).transpose(0, 3, 1, 2)

        self.X = X
        self.col = col
        self.col_W = col_W

        return Z

    def backward(self, dout):
        FN, C, FL, FL = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.dB = np.sum(dout, axis=0)
        self.dW = self.col.T @ dout
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FL, FL)

        dcol = dout @ self.col_W.T
        dx = col2im(dcol, self.X, FL, self.Stride, self.pad)

        return dx


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, X):
        self.mask = (X <= 0)  # 定义数组小于等于0为True， 反之则为false
        out = X.copy()
        out[self.mask] = 0  # 当mask为True则为0
        return out

    def backward(self, dout):
        dout[self.mask] = 0  # 当mask为True，dout则为0
        dx = dout

        return dx


class Pooling:
    def __init__(self, K=2, Stride=2, pad=0):
        self.K = K  # 池化核的大小
        self.Stride = Stride  # 池化步长
        self.pad = pad

        # 以下均作为方向传播的参数作为记录，确定方向传播的变量进行存储
        self.X = None
        self.arg_max = None

    '''
    前向传播，首先过一遍im2col，得到展开的图像在把通道忽略进行堆叠，
    分别使用argmax确定索引
    max确定最大值
    '''

    def forward(self, X):
        n, C, L, L = X.shape
        if L % 2 != 0:
            self.pad = 1
        L_out = int((L + self.pad - self.K) / self.Stride + 1)

        col = im2col(X, self.K, self.pad, self.Stride)
        col = col.reshape(-1, self.K * self.K)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(n, L_out, L_out, C).transpose(0, 3, 1, 2)

        self.X = X
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.K * self.K
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten() # np.arange(self.arg_max)表示的是创建索引数组，self.arg_amx.flatten()就是将其展开平铺，得到了索引，其对应的值就是dout的平铺
        dmax = dmax.reshape(dout.shape + (pool_size,))  # 此处表示dmax的reshape成原本dout的大小外加一个4的大小用来表示，取了哪个作为最大值

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] *dmax.shape[2], -1)
        dx = col2im(dcol, self.X, self.K, self.Stride, self.pad)  # 将col数组转换会img

        return dx

class Fullconnect:
    def __init__(self, W, B):
        self.W = W
        self.B = B

        self.originX_shape = None  # 存储变换之前的X
        self.X = None  # 存储展开之后的X

        self.dW = None
        self.dB = None

    def forward(self, X):
        self.originX_shape = X.shape
        n = X.shape[0]
        X = X.reshape(n, -1)
        self.X = X
        Z = X @ self.W + self.B

        return Z

    def backward(self, dout):
        dx = dout @ self.W.T
        self.dW =  self.X.T @ dout
        self.dB = np.sum(dout, axis=0)

        dx = dx.reshape(*self.originX_shape)
        return dx


class Softmax:
    def __init__(self):
        self.Y = None
        self.loss = None
        self.t = None

    def forward(self, X, Y):
        self.Y = softmax(X)
        self.t = Y
        self.loss = cost(self.Y, Y)

        return self.loss

    def backward(self, dout = 1):
        batch_size = self.t.shape[0]
        dx = (self.Y - self.t) / batch_size

        return dx


class ConvNet:
    '''
    input_dim 单个图片的维度 1通道 20*20大小
    conv_param参数
    '''
    def __init__(self, input_dim=(1, 20, 20),
                 conv_param = {'filter_num':10,'pad':0,'stride':1,'filter_size':3},
                 output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num'] # 卷积核数
        filter_size = conv_param['filter_size'] # 卷积核大小
        filter_pad = conv_param['pad'] # 补零数
        filter_stride = conv_param['stride'] # 步长
        input_size = input_dim[1] # 输入图片的大小
        conv_output_size = (input_size - filter_size + 2*filter_pad)/filter_stride+1 # 卷积层输出的大小
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2)) # 池化层大小

        # 权重初始化 涉及到W，B的地方 分别是卷积， 全连接
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['B1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(pool_output_size, output_size)
        self.params['B2'] = np.zeros(output_size)

        # 每一层的初始化
        self.layers = {}
        self.layers['Conv'] = Conv(self.params['W1'], self.params['B1'],
                                   conv_param['stride'], conv_param['pad'])
        self.layers['Relu'] = Relu()
        self.layers['Pool'] = Pooling(K=2, Stride=2, pad=0)
        self.layers['Fullconnect'] = Fullconnect(self.params['W2'], self.params['B2'])
        self.last_layer = Softmax()
        self.c = 0
        self.lr = 0.01

    def loss(self, X, Y):
        X = self.predict(X)

        return self.last_layer.forward(X, Y)

    def predict(self, X):
        for layer in self.layers.values():
            X = layer.forward(X)

        return X

    def gradient(self, X, Y):
        self.loss(X, Y)

        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'], grads['B1'] = self.layers['Conv'].dW, self.layers['Conv'].dB
        grads['W2'], grads['B2'] = self.layers['Fullconnect'].dW, self.layers['Fullconnect'].dB

        return grads

    def update(self, X, Y, i):
        grads = self.gradient(X, Y)
        self.lr = cosine_decay_with_warmup(i, 0.03, 1000, self.lr, 200, 0)
        optimizer = Adam(self.lr)
        self.params = optimizer.update(self.params, grads)


class Adam:

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

        return params


data = np.load("data.npy")
np.random.shuffle(data)
X = data[:, :-1].reshape(data.shape[0], 1, 20, 20).transpose(0, 1, 3, 2)
Y = data[:, -1].astype(np.int32)
(n, _, L, _) = X.shape
Y = onehotEncoder(Y, 10)
iterations = 5000
net = ConvNet(input_dim=(1, 20, 20),
              conv_param={'filter_num': 10, 'pad': 0, 'stride': 1, 'filter_size': 3},
              output_size=10, weight_init_std=0.1)
acc = 0
for i in range(iterations):
    '1'
    X1 = X[i%5*1000:(i%5+1)*1000]
    Y1 = Y[i%5*1000:(i%5+1)*1000]
    '1'
    net.update(X1, Y1, i)
    #c = net.loss(X, Y)
    Y_hat = net.last_layer.Y
    #print(c)
    acc += test(Y_hat, Y1, i)
    if i % 5 == 0:
        print(acc/5)
        acc = 0

    #(acc)
