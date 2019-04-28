

# try to back pre git

import numpy as np


def MeanNormalize(OrigArr):
    pass

def sigmoid(inX):
    return 1.0/(1 + np.exp(-inX))

def FeatureScaling(OrigArr):
    # Preprocessing

    n_examples = OrigArr.shape[0]
    n_features = OrigArr.shape[1]

    mean = np.zeros(n_features)
    std = np.zeros(n_features)

    StdMeanNpArr = np.zeros((n_examples, n_features))

    mean = np.average(OrigArr, axis=0).reshape(1, n_features)
    std = np.std(OrigArr, axis=0).reshape(1, n_features)
    StdMeanNpArr = (OrigArr - mean) / std

    return StdMeanNpArr, mean, std

def TestScaling(OrigArr, mean, std):

    return (OrigArr - mean) / std

def TXTtoNumpy(TXTfilename, lableState=True, Print=False, delim = '\t'):
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
        labels = [int(float(line[n_features])) for line in stringArr]
    else:
        n_features = len(stringArr[0])

    if Print:
        print("n_examples: ", n_examples)
        print("n_features: ", n_features)

    floatList = np.zeros((n_examples, n_features))

    for i in range(0, n_features):
        floatList[:,i] = [float(line[i]) for line in stringArr]

    labels = np.array(labels).reshape(n_examples, 1)

    if lableState:
        return floatList, labels
    else:
        return floatList

