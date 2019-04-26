
# logistic回归


import numpy as np
import matplotlib.pyplot as plt


def DataSet(X_orig, Y_orig):

    '''

    :param X_orig:
    :param Y_orig:
    :return: X: shape=(m_examples, n_features+1)
             Y: shape=(m_examples, 1)
    '''
    mExamples = X_orig.shape[0]
    nFeatures = X_orig.shape[1]

    # 训练数据中添加一列为全0，作为x0，theta0的参数,
    # 训练集的维度为m*（n+1）
    trainX = np.zeros((mExamples, nFeatures + 1))
    trainX[:, 0] = np.ones(mExamples)
    trainX[:, 1:] = X_orig

    # 定义标签数组
    trainY = np.zeros((mExamples, 1))
    trainY[:, 0] = Y_orig

    return trainX, trainY



def logisticRegression(X, Y, iteration=1000, learning_rate=0.1, regularization=0):

    '''

    :param X: shape=(m_examples, n_features+1)
    :param Y: shape=(m_examples, 1)
    :param iteration:
    :return: parameters, shape=(n_features+1, 1)
    '''
    # 读取样本个数，特征个数
    m_examples = X.shape[0]
    n_features = X.shape[1 ] -1

    # 初始化参数
    parameters = np.zeros((n_features +1, 1))

    # 开始迭代
    iter_i = 0
    while iter_i <= iteration:
        iter_i = iter_i +1

        Z = np.dot(X, parameters)
        HZ = 1.0 / (1.0 + np.exp(-Z))

        # 计算代价函数(此处不加正则化项，只需要在求梯度的时候加入即可)
        J_cost = np.sum(np.multiply(Y, np.log(HZ)) + np.multiply((1 - Y), np.log(1 - HZ))) / (-m_examples)

        # 计算梯度
        PartDeri = np.dot(X.T, (HZ - Y)) / m_examples

        if regularization > 0:
            PartDeri = PartDeri + (regularization * parameters) / m_examples

        # 更新参数
        parameters = parameters - learning_rate * PartDeri
        parameters = parameters.reshape(n_features + 1, 1)

        print("[", iter_i, "] [cost] ", J_cost)

    return parameters


def predict_all(parameters, X, Y):
    Z = np.dot(X, parameters)
    # HZ = 1.0 / (1.0 + np.exp(-Z))

    m_examples = Z.shape[0]

    i = 0
    while i < m_examples:
        if Z[i] > 0:
            Z[i] = 1
        else:
            Z[i] = 0;
        i = i + 1

    result = np.sum(abs(Z - Y))
    accurancy = 1 - result / m_examples

    return accurancy


if __name__ == '__main__':
    # 创建数据集合
    X_orig = np.arange(0, 10, 1).reshape(10, 1)
    print("X_orig:", X_orig)

    Y_orig = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    print("Y_orig: ", Y_orig)

    X, Y = DataSet(X_orig, Y_orig)
    print("X:", X)
    print("Y", Y)

    parameters = logisticRegression(X, Y, regularization=0)

    accurancy = predict_all(parameters, X, Y)
    print("train accrancy:", accurancy)

