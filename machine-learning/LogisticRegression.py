


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



def gradDecent(origData, labels, learning_rate=0.001, n_iterations=2000, lamda=1.0, show_cost=True):

    #特征缩放
    npData = np.ones((origData.shape[0], origData.shape[1]+1), dtype=float)
    npData[:,1:] = FeatureScaling(origData)

    n_examples = npData.shape[0]
    n_features = npData.shape[1]

    #将权重随机初始化，包括W0项，其X0 = 1
    Weights = np.random.rand(n_features, 1)

    Cost = np.ones(n_iterations)

    for k in range(0, n_iterations):

        temp = np.dot(npData, Weights)
        h = sigmoid(temp)
        error = (h - labels)
        # 计算损失值
        Cost[k] = -np.sum(np.multiply(labels, np.log(h)) + np.multiply((1.0 - labels), np.log(1.0 - h)))/n_examples
        # 加上正则化，防止过拟合
        Weights[0,:] = Weights[0,:] - learning_rate * np.dot(npData[:,0].T, error)
        Weights[1:,:] = Weights[1:,:] * lamda - learning_rate * np.dot(npData[:,1:].T, error)

    if show_cost:
        plt.plot(range(0, n_iterations), Cost)
        plt.show()

    # 绘制图
    plotBestFit(npData, labels, Weights)

    accuracy = Accuracy(npData, labels, Weights)

    return Weights

def plotBestFit(npData, labels, Weight):

    n_examples = npData.shape[0]

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


if __name__ == '__main__':

    filename = './ML-SourceCode/Ch05/testSet.txt';

    origData, labels = TXTtoNumpy(filename, lableState=True, Print=True)

    Weights = gradDecent(origData, labels, show_cost=False)


    print(Weights)