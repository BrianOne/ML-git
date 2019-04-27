


# git branch LR
from tools import  *
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(inX):
    return 1.0/(1 + np.exp(-inX))

def Accuracy(npData, labels, Weights):

    n_examples = npData.shape[0]

    n_right = 0

    for i in range(0, n_examples):

        h = sigmoid(np.dot(npData[i,:], Weights))

        if h >= 0.5:
            Ypredict = 1
        else:
            Ypredict = 0

        if Ypredict == labels[i,:]:
            n_right += 1

    accuracy = float(n_right)/float(n_examples)

    print("Accuracy: ", accuracy*100, "%")

def plotBestFit(npData, labels, Weight):

    n_examples = npData.shape[0]
    n_features = npData.shape[1]

    if n_features == 3:
        X0 = []; Y0 = []
        X1 = []; Y1 = []

        for i in range(0, n_examples):
            if labels[i,:] == 0:
                X0.append(npData[i, 1]); Y0.append(npData[i, 2])
            else:
                X1.append(npData[i, 1]); Y1.append(npData[i, 2])

        x = np.arange(-5.0, 5, 0.1)
        y = -(Weight[0] + Weight[1]*x)/Weight[2]


        plt.scatter(X0, Y0, c='red', marker='^')
        plt.scatter(X1, Y1, c='blue', marker="o")
        plt.plot(x, y)
        plt.show()

# 批量梯度下降算法
def BatchGraDescent(npData, labels, Weights, learning_rate, lamda):

    n_examples = npData.shape[0]

    temp = np.dot(npData, Weights)
    h = sigmoid(temp)
    error = (h - labels)
    # 计算损失值
    Cost = -np.sum(np.multiply(labels, np.log(h)) + np.multiply((1.0 - labels), np.log(1.0 - h))) / n_examples
    # 加上正则化，防止过拟合
    Weights[0, :] = Weights[0, :] - learning_rate * np.dot(npData[:, 0].T, error)
    Weights[1:, :] = Weights[1:, :] * lamda - learning_rate * np.dot(npData[:, 1:].T, error)

    return Weights, Cost

# 随机梯度下降算法
def StocGraDescent(npData, labels, Weights, learning_rate, lamda):

    n_examples = npData.shape[0]
    n_features = npData.shape[1]

    Cost = 0

    WeightsArr = np.zeros((n_examples, n_features))

    for i in range(0, n_examples):

        x = npData[i:i+1,:]
        y = labels[i,:]

        learning_rate = 4/(1.0 + i) +0.01

        temp = np.dot(x, Weights)
        h = sigmoid(temp)
        error = (h - y)
        Cost += -(y * np.log(h) + (1 - y) * np.log(1 - h))
        Weights[0,:] = Weights[0,:] - learning_rate * x[:,0] * error
        temp1 = np.multiply(x[1:], error)
        Weights[1:,:] = Weights[1:,:] * lamda - learning_rate * x[:,1:].T * error

        WeightsArr[i:i+1, :] = Weights.T

    Cost = Cost/n_examples

    return WeightsArr, Cost, Weights


def LR(origData, labels, learning_rate=0.001, n_iterations=200, lamda=1.0, show_cost=True,
                    show_boundry = False):

    #特征缩放
    npData = np.ones((origData.shape[0], origData.shape[1]+1), dtype=float)
    npData[:,1:] = FeatureScaling(origData)

    n_examples = npData.shape[0]
    n_features = npData.shape[1]

    #将权重随机初始化，包括W0项，其X0 = 1
    Weights = np.random.rand(n_features, 1)

    Cost = np.ones(n_iterations)

    WeightsArr = np.zeros((n_iterations*n_examples, n_features))

    for k in range(0, n_iterations):

        # Weights, Cost[k] = BatchGraDescent(npData, labels, Weights, learning_rate, lamda)
        kWeightsArr, Cost[k], Weights = StocGraDescent(npData, labels, Weights, learning_rate, lamda)
        WeightsArr[k*n_examples:k*n_examples+n_examples,:] = kWeightsArr
        pass

    if show_cost:
        plt.plot(range(0, n_iterations), Cost)
        plt.show()

    # 绘制边界图
    if show_boundry:
        plotBestFit(npData, labels, Weights)

    accuracy = Accuracy(npData, labels, Weights)

    plt.figure(311)
    plt.plot(range(0, n_iterations*n_examples), WeightsArr[:, 0])
    plt.figure(312)
    plt.plot(range(0, n_iterations*n_examples), WeightsArr[:, 1])
    plt.figure(313)
    plt.plot(range(0, n_iterations*n_examples), WeightsArr[:, 2])
    plt.show()

    return Weights


if __name__ == '__main__':

    filename = './ML-SourceCode/Ch05/testSet.txt'

    origData, labels = TXTtoNumpy(filename, lableState=True, Print=True)

    Weights = LR(origData, labels, show_cost=True, show_boundry=True)


    print(Weights)