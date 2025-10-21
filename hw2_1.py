import operator
from math import log


def createDataSet():
    dataSet = [['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
               ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],
               ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
               ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],
               ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
               ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '是'],
               ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '是'],
               ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '是'],
               ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '否'],
               ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '否'],
               ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '否'],
               ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '否'],
               ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '否'],
               ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '否'],
               ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '否'],
               ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '否'],
               ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '否']]
    features = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
    return dataSet, features

    #TODO: 计算数据集的信息熵，通过读取和遍历数据集中的类别标签，计算每个类别标签的概率，并计算总的信息熵，返回该数据集的熵值。
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

    #TODO：调用calcShannonEnt函数计算数据集的熵，获取所有的特征值，再遍历每个特征值，调用splitDataSet函数划分数据集，计算划分后子数据集的熵值，再计算信息增益，最后返回信息增益最大的特征索引。
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1

    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)

        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)

        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, features):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = features[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (features[bestFeat])

    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)

    for value in uniqueVals:
        subFeatures = features[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subFeatures)

    return myTree


if __name__ == '__main__':
    dataSet, features = createDataSet()
    print(createTree(dataSet, features))