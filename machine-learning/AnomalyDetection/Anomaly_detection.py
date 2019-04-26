
import numpy as np
import random
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

def GaussianParamEstimation(npArr, GaussianType = 'Normal'):

    '''

    :param npArr: shape=(n_examples, n_features)
    :param GaussianType: 'Normal' or 'Multi'
    :return:
    '''

    n_features = npArr.shape[1]

    # mean = np.zeros(n_features)
    mean = np.average(npArr, axis=0)

    if GaussianType == 'Normal':
        # std = np.zeros(n_features)
        std = np.std(npArr, axis=0)

        return mean, std

    elif GaussianType == 'Multi':
        sigma = np.cov(npArr - mean, rowvar=0)

        return mean, sigma

def NormalGaussion(X, mean, std):

    '''

    :param X: shape=(1, n_features)
    :param mean: shape=(1, n_features)
    :param std: shape=(1, n_features)
    :return:
    '''

    n_feature = X.shape[1]

    P = 1;

    for i in range(0,n_feature):
        temp1 = ( 1 / (np.sqrt(2*np.pi) * std[i]))
        temp2 = np.exp( -pow(X[:,i] - mean[i], 2) / (2 * pow(std[i],2)))
        P = P * (temp1 * temp2)

    return P

def MultiGaussion(X, mean, sigma):

    '''
    :param X: shape=(1, n_features)
    :param mean:  shape=(1, n_features)
    :param sigma: shape=(n_features, n_features)
    :return:
    '''

    temp1 = ( 1 / (pow(2*np.pi, np.pi/2) * np.sqrt(np.linalg.det(sigma))))
    temp2 = np.dot((X-mean), np.linalg.inv(sigma))
    temp3 = np.exp( (-1/2) * np.dot(temp2, (X-mean).T))
    P = temp1 * temp3

    return P

def AnomalyDetection(npArr, labels, iterations, lamda_step=0.001, lamda=0.001):

    '''

    :param npArr: shape=(n_examples, n_features)
    :param labels:  shape=(n_examples, 1)
    :param iterations:
    :param lamda_step:
    :param lamda:
    :return:
    '''

    n_examples = npArr.shape[0]
    n_features = npArr.shape[1]

    # 将labels的列表类型转为numpy类型
    labels = np.array(labels).reshape(n_examples, 1)

    # 找出标记为非0（异常样本）的索引
    anomalyIndex = []
    for i in range(0, n_examples):
        if(labels[i:i+1,:] != 0):
            anomalyIndex.append(i)

    # 根据异常样本索引得到异常数据和异常标记
    anomalyArr = npArr[anomalyIndex, :]
    anomalyLabels = labels[anomalyIndex, :]

    # 获得异常样本的测试数据
    n_anomaly = anomalyArr.shape[0]
    n_anomalyTest = int(n_anomaly/2)
    anomalyTestArr = anomalyArr[0:n_anomalyTest, :]
    anomalyTestLabels = anomalyLabels[0:n_anomalyTest, :]

    # 获得异常样本的验证数据
    anomalyDevArr = anomalyArr[n_anomalyTest:, :]
    anomalyDevLabels = anomalyLabels[n_anomalyTest:, :]

    # 去掉异常数据得到正常数据
    NormalArr = np.delete(npArr, anomalyIndex, axis=0)
    NormalLabels = np.delete(labels, anomalyIndex ,axis=0)

    # 样本数更新为正常样本数量
    n_examples = NormalArr.shape[0]

    # 正常样本中测试数据和验证数据集的大小
    n_test = int(n_examples * 0.2)
    n_dev = int(n_examples * 0.2)

    # 划分训练数据，验证数据，测试数据
    testIndex = random.sample(range(0, n_examples), n_test) # 获得测试数据索引
    NormalTestArr = NormalArr[testIndex, :] # 获得正常样本的测试数据
    NormalTestLabels = NormalLabels[testIndex, :] # 获得正常样本的测试数据标签

    # 训练数据+验证数据，用于交叉验证
    delNormalTestArr = np.delete(NormalArr, testIndex, axis=0)
    delNormalTestLabels = np.delete(NormalLabels, testIndex, axis=0)

    # 在正常样本中，去掉测试数据，得到其余数据的大小
    n_delNormalTest = delNormalTestArr.shape[0]

    # 存储各lamda值，和其对应的F1值
    lamdaToF1 = np.zeros((iterations, 2))

    # 不断更新lamda值
    for iter in range(0, iterations):

        # 交叉验证10次
        F1Arr = np.zeros(10)

        for cross_i in range(0,10):
            # 获得正常样本的验证数据
            devNormalIndex = random.sample(range(0, n_delNormalTest), n_dev) # 获得正常样本的验证数据索引
            devNormalArr = delNormalTestArr[devNormalIndex, :] # 获得正常样本的验证数据
            devNormalLabels = delNormalTestLabels[devNormalIndex, :] # 获得验证数据标签

            # 获得正常样本的训练数据
            trainArr = np.delete(delNormalTestArr, devNormalIndex, axis=0) # 获得训练数据
            mean, std = GaussianParamEstimation(trainArr, GaussianType='Multi')

            # 正常样本为负样本，异常样本为正样本
            # 检查正常样本验证数据的效果
            F1Arr[cross_i] = computeF1(anomalyDevArr, devNormalArr, mean, std, lamda, GaussionType=MultiGaussion)

        # 计算交叉验证的平均F1值
        F1 = np.average(F1Arr)
        lamdaToF1[iter, 0] = lamda
        lamdaToF1[iter, 1] = F1

        # 更新lamda
        lamda = lamda + lamda_step

    MaxF1Index = np.argmax(lamdaToF1[:, 1])
    lamda = lamdaToF1[MaxF1Index, 0]

    # 检测测试数据效果
    F1_test = computeF1(anomalyTestArr, NormalTestArr, mean, std, lamda, GaussionType=MultiGaussion)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(trainArr[:,0], trainArr[:,1], marker='^', c='red')
    ax.scatter(anomalyTestArr[:,0], anomalyTestArr[:,1], marker='o', c='blue')
    plt.show()

    return mean, std, lamda

def computeF1(PData, NData, mean, std, lamda, GaussionType = NormalGaussion):

    '''

    :param PData: Positive category
    :param NData: Negative category
    :param mean:
    :param std: std(NormalGaussion) or sigma(MultiGaussion)
    :param lamda: Min probability
    :param GaussionType: NormalGaussion or MultiGaussion
    :return: F1 combine Precision and Recall
    '''

    n_PData = PData.shape[0]
    n_NData = NData.shape[0]

    TP = FP = FN = TN = 0.0
    for i in range(0, n_PData):
        P = GaussionType(PData[i:i+1, :], mean, std)
        if P < lamda:  # True Positive
            TP += 1.0
        elif P >= lamda:  # False Negative
            FN += 1.0

    for i in range(0, n_NData):
        P = GaussionType(NData[i:i+1, :], mean, std)
        if P >= lamda:  # True Negative
            TN += 1.0
        elif P < lamda:  # False Positive
            FP += 1.0

    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = 2 * Precision * Recall / (Precision + Recall)

    return F1


if __name__ == '__main__':

    filename = './testSet3.txt'
    npArr, labels = TXTtoNumpy(filename, lableState=True, Print=True)
    mean, std, lamda = AnomalyDetection(npArr, labels, iterations=10)




