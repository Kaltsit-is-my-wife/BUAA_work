import numpy as np
from os import listdir


# KNN主体算法
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        label = labels[sortedDistIndicies[i]]
        classCount[label] = classCount.get(label, 0) + 1
    maxClassCount = max(classCount, key=classCount.get)
    return maxClassCount


# 数据预处理：将32x32图像文件转为1x1024向量
def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    fr.close()  # 显式关闭文件，避免资源泄漏
    return returnVect


#TODO:从目标文件夹中获取测试数据集和训练数据集，进行转换后再进行计算，利用KNN算法进行分类，并统计错误率，最后输出结果。
def handwritingClassTest():
    hwLabels = []  # 存储训练标签
    # 1. 加载训练数据
    trainingFileList = listdir('data/trainingDigits')  # 获取训练文件列表
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))  # 初始化训练矩阵

    for i in range(m):
        fileNameStr = trainingFileList[i]
        # 从文件名提取标签，如 "0_13.txt" -> 0
        classNumStr = int(fileNameStr.split('_')[0])
        hwLabels.append(classNumStr)
        # 转换图像为向量并存入矩阵
        trainingMat[i, :] = img2vector('data/trainingDigits/%s' % fileNameStr)

    # 2. 测试阶段
    testFileList = listdir('data/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)

    for i in range(mTest):
        fileNameStr = testFileList[i]
        realLabel = int(fileNameStr.split('_')[0])  # 真实标签
        # 转换测试图像
        testVec = img2vector('data/testDigits/%s' % fileNameStr)
        # KNN分类（k=3）
        classifierResult = classify0(testVec, trainingMat, hwLabels, 3)
        if classifierResult != realLabel:
            errorCount += 1.0

    # 3. 输出结果
    print("错误总数: %d" % int(errorCount))
    print("测试样本数: %d" % mTest)
    print("错误率: %.2f%%" % (errorCount / mTest * 100))


# 主程序入口
if __name__ == '__main__':
    handwritingClassTest()