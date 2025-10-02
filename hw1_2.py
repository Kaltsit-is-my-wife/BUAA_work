from numpy import *
from matplotlib.font_manager import FontManager
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score


# TODO定义读取文件的函数，参数为文件名
def file2matrix(filename):
    # TODO打开和读取文件内容，写到fr变量里
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    # TODO
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    # TODO
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    # TODO
    return returnMat, classLabelVector


# TODO
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    # TODO
    normDataSet = dataSet - tile(minVals, (m, 1))
    # TODO
    normDataSet = normDataSet / tile(ranges, (m, 1))
    # TODO
    return normDataSet, ranges, minVals


# TODO
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # TODO
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # TODO
    sqDiffMat = diffMat ** 2
    # TODO
    sqDistances = sqDiffMat.sum(axis=1)
    # TODO
    distances = sqDistances ** 0.5
    # TODO
    sortedDistIndicies = distances.argsort()
    classCount = {}
    # TODO
    for i in range(k):
        label = labels[sortedDistIndicies[i]]
        classCount[label] = classCount.get(label, 0) + 1
    # TODO
    maxClassCount = max(classCount)
    return maxClassCount


# TODO
def datingClassTest():
    # TODO
    testData, testLabels = file2matrix('knn_testData.txt')
    trainData, trainLabels = file2matrix('knn_trainData.txt')

    # TODO
    testNorm, _r, _m = autoNorm(testData)
    trainNorm, _rr, _mm = autoNorm(trainData)
    errorCount = 0.0
    accuracy = []
    # TODO
    for kk in range(1, 31):
        for i in range(testNorm.shape[0]):
            result = classify0(testNorm[i], trainNorm, trainLabels, kk)
            if (result != testLabels[i]):
                errorCount += 1.0
        # TODO
        accuracy.append(100 - errorCount / float(testNorm.shape[0]) * 100)
        # TODO
        errorCount = 0.0

    # TODO
    plt.plot(range(1, 31), accuracy)
    # TODO
    plt.xlabel('K')
    # TODO
    plt.ylabel('准确率(%)')
    # TODO
    plt.show()

fm = FontManager()
mat_fonts = set(f.name for f in fm.ttflist)
plt.rcParams['font.sans-serif'] = ['SimHei']

# TODO:
if __name__ == '__main__':
    datingClassTest()