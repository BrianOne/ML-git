


# git branch LR
from tools import  *
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(inX):
    return 1.0/(1 + np.exp(-inX))

def gradDecent(origData, labels, learning_rate=0.001, n_iterations=20, lamda=0.1, show_cost=True):

    #特征缩放
    npData = FeatureScaling(origData)

    n_examples = npData.shape[0]
    n_features = npData.shape[1]

    #将权重随机初始化
    Weights = np.random.rand(n_features, 1)

    Cost = np.ones(n_iterations)

    for k in range(0, n_iterations):

        temp = np.dot(npData, Weights)
        h = sigmoid(temp)
        error = (h - labels)
        # 计算损失值
        Cost[k] = -np.sum(np.multiply(labels, np.log(h)) + np.multiply((1.0 - labels), np.log(1.0 - h)))/n_examples
        # 加上正则化，防止过拟合
        Weights = Weights * lamda - learning_rate * np.dot(npData.T, error)

    if show_cost:
        plt.plot(range(0, n_iterations), Cost)
        plt.show()

    return Weights

def plotBestFit(npData, Weight):

    n_examples = npData.shape[0]



if __name__ == '__main__':

    filename = './ML-SourceCode/Ch05/testSet.txt';

    npData, labels = TXTtoNumpy(filename, lableState=True, Print=True)

    Weight = gradDecent(npData, labels, show_cost=False)

    print(Weight)