

import numpy as np
import matplotlib.pyplot as plot


def DataDownload():
    '''

    :return:X-shape=(m_examples, n_features)
            Y-shape=(m_examples, 1)
    '''
    X_orig = np.arange(0, 10, 1).reshape(10, 1)
    Y_orig = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    return X_orig, Y_orig

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

class CNN():
    pass



if __name__ == '__main__':

    # 读取数据
    X_orig, Y_orig = DataDownload()

    # X-shape=(m_examples, n_features+1)
    # Y-shape=(m_examples, 1)
    trainX, trainY = DataSet(X_orig, Y_orig)

    # 遍历m个所有数据
    for a_1 in trainX:
        # a_1.shape=(1, n_features+1)



