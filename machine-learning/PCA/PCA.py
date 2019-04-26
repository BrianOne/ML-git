

import numpy as np
import matplotlib.pyplot as plt

def TXTtoNumpy(TXTfilename, lableState=False, Print=False, delim = '\t'):
    '''

    :param TXTfilename: Path about TXT file
    :param lableState: True for have labels of data
    :param print: to print info about data
    :param delim: to split '\t'
    :return:
    '''
    TXTfr = open(TXTfilename)
    TXTList = TXTfr.readlines()
    stringArr = [line.strip().split(delim) for line in TXTList]


    n_examples = len(stringArr)

    if lableState:
        n_features = len(stringArr[0])-1
        labels = np.zeros(n_examples)
        labels = [int(line[n_features]) for line in stringArr]
    else:
        n_features = len(stringArr[0])

    if Print:
        print("n_examples: ", n_examples)
        print("n_features: ", n_features)

    floatList = np.zeros((n_examples, n_features))

    for i in range(0, n_features):
        floatList[:,i] = [float(line[i]) for line in stringArr]

    if lableState:
        return floatList, labels
    else:
        return floatList

def pca(npArr, k, show = False):

    '''

    :param npArr: shape=(n_examples, n_features)
    :param k: to keep k components
    :param show: True to show figure about origData and reconData
    :return: LowNpArr, loss
    '''
    # Preprocessing

    n_examples = npArr.shape[0]
    n_features = npArr.shape[1]

    mean = np.zeros(n_features)
    std = np.zeros(n_features)

    StdMeanNpArr = np.zeros((n_examples, n_features))

    mean = np.average(npArr, axis=0).reshape(1,n_features)
    std = np.std(npArr, axis=0).reshape(1,n_features)
    StdMeanNpArr = (npArr - mean) / std

    # pca
    sigma = np.cov(StdMeanNpArr, rowvar=0)
    eigValue, eigVects = np.linalg.eig(sigma) # 获得协方差矩阵的特征值，特征向量
    eigValInd = np.argsort(eigValue) # 返回特征值从小到大排序的索引
    eigValInd = eigValInd[:-(k+1):-1] # 从后向前一共取k个值的索引
    redEigVects = eigVects[:,eigValInd] # 取出指定索引的特征向量
    LowNpArr = np.dot(StdMeanNpArr, redEigVects) # 原数据与选定的特征向量内积，得到降维数据
    reconNpArr = np.dot(LowNpArr, redEigVects.T) # 重构数据

    redEigValue = eigValue[eigValInd]
    loss = 1 - np.sum(redEigValue)/np.sum(eigValue) # 重构数据的损失程度
    print("PCA loss: ", loss)

    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(StdMeanNpArr[:, 0], StdMeanNpArr[:, 1], marker = '^', c='red')
        ax.scatter(reconNpArr[:,0],reconNpArr[:,1], marker='o', c='blue')
        plt.show()

    return LowNpArr, loss




if __name__ == '__main__':
    filename = './testSet3.txt'
    npArr, lables = TXTtoNumpy(filename, lableState=True, Print=True)
    LowNpArr, loss = pca(npArr, k=1, show=True)













