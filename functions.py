import numpy as np


# Utilities
def onehotEncoder(Y, ny):
    return np.eye(ny)[Y]


# Compute the cost function
def cost(Y_hat, Y):
    n = Y.shape[0]
    c = -np.sum(Y * np.log(Y_hat)) / n

    return c


def test(Y_hat, Y, i):
    Y_out = np.zeros_like(Y)

    idx = np.argmax(Y_hat, axis=1)
    Y_out[range(Y.shape[0]), idx] = 1
    acc = np.sum(Y_out * Y) / Y.shape[0]
    print("%d:Training accuracy is: %f" % (i,acc))

    return acc


def im2col(X, K, pad, S):
    n, C, L, L = X.shape
    L_out = int((L - K + pad) / S + 1)
    img = np.pad(X, [(0, 0), (0, 0), (0, pad), (0, pad)], 'constant')  # 进行补零操作
    col = np.zeros((n, C, K, K, L_out, L_out))

    for y in range(K):
        y_max = y + S * L_out
        for x in range(K):
            x_max = x + S * L_out
            col[:, :, y, x, :, :] = img[:, :, y:y_max:S, x:x_max:S]  # 选取了y，x对应的所框选的img中的位置存到col中

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(n * L_out * L_out, -1)  # 进行reshape每一个对应一个Lout*Lout
    return col


def col2im(col, input_shape, K, S=1, pad=0):
    N, C, L, L = input_shape.shape
    L_out = int((L + pad - K) // S + 1)
    col = col.reshape(N, L_out, L_out, C, K, K).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, L + pad + S - 1, L + pad + S - 1))
    for y in range(K):
        y_max = y + S * L_out
        for x in range(K):
            x_max = x + S * L_out
            img[:, :, y:y_max:S, x:x_max:S] += col[:, :, y, x, :, :]

    return img[:, :, pad:L + pad, pad:L + pad]


def Relu(Z):
    return np.maximum(0, Z)


def softmax(Z):
    Z = Z.T
    Z = Z - np.max(Z, axis=0)
    Y = np.exp(Z) / np.sum(np.exp(Z), axis=0)

    return Y.T
