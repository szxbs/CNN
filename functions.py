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
    #print("%d:Training accuracy is: %f" % (i,acc))

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


def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
    '''
    参数：
        global_step: 上面定义的Tcur，记录当前执行的步数。
        learning_rate_base：预先设置的学习率，当warm_up阶段学习率增加到learning_rate_base，就开始学习率下降。
        total_steps: 是总的训练的步数，等于epoch*sample_count/batch_size,(sample_count是样本总数，epoch是总的循环次数)
        warmup_learning_rate: 这是warm up阶段线性增长的初始值
        warmup_steps: warm_up总的需要持续的步数
        hold_base_rate_steps: 这是可选的参数，即当warm up阶段结束后保持学习率不变，知道hold_base_rate_steps结束后才开始学习率下降
    '''
    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')
    #这里实现了余弦退火的原理，设置学习率的最小值为0，所以简化了表达式
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(np.pi *
        (global_step - warmup_steps - hold_base_rate_steps) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    #如果hold_base_rate_steps大于0，表明在warm up结束后学习率在一定步数内保持不变
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        #线性增长的实现
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        #只有当global_step 仍然处于warm up阶段才会使用线性增长的学习率warmup_rate，否则使用余弦退火的学习率learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)

    return np.where(global_step > total_steps, 0.0, learning_rate)

