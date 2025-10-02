import numpy as np
from matplotlib.font_manager import FontManager
import matplotlib.pyplot as plt
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split, cross_val_score


def file2matrix(filename):
    with open(filename, 'r', encoding='utf-8') as fr:
        lines = [line.strip() for line in fr if line.strip()]
    numberOfLines = len(lines)

    dataMat = np.zeros((numberOfLines, 3), dtype=float)
    labelVec = []
    for idx, line in enumerate(lines):
        parts = line.split('\t')
        dataMat[idx, :] = list(map(float, parts[0:3]))
        labelVec.append(int(parts[-1]))
    return dataMat, labelVec


def autoNorm(dataSet):
    minVals = dataSet.min(axis=0)
    maxVals = dataSet.max(axis=0)
    ranges = maxVals - minVals

    # 避免除以 0：将范围为 0 的特征置为 1（该特征归一化后恒为 0）
    ranges_safe = ranges.copy()
    ranges_safe[ranges_safe == 0] = 1.0

    normDataSet = (dataSet - minVals) / ranges_safe
    return normDataSet, ranges_safe, minVals


def classify0(inX, dataSet, labels, k):
    dataSet = np.asarray(dataSet)
    labels = np.asarray(labels)
    dataSetSize = dataSet.shape[0]
    k = min(k, dataSetSize)

    # 距离
    diffMat = dataSet - inX  # (N, D)
    distances = np.sqrt((diffMat ** 2).sum(axis=1))
    sortedIdx = distances.argsort()

    # 多数表决
    classCount = {}
    for i in range(k):
        lab = int(labels[sortedIdx[i]])
        classCount[lab] = classCount.get(lab, 0) + 1

    # 返回出现次数最多的类别；如有并列，用距离最近的作为决胜
    maxCount = max(classCount.values())
    candidates = [lab for lab, cnt in classCount.items() if cnt == maxCount]
    if len(candidates) == 1:
        return candidates[0]
    for i in sortedIdx[:k]:
        lab = int(labels[i])
        if lab in candidates:
            return lab


def datingClassTest():
    testData, testLabels = file2matrix('data/knn_testData.txt')
    trainData, trainLabels = file2matrix('data/knn_trainData.txt')

    # 只用训练集统计量做归一化，并将其应用在测试集上
    trainNorm, ranges, minVals = autoNorm(trainData)
    testNorm = (testData - minVals) / ranges

    accuracy = []
    for kk in range(1, 31):
        errors = 0
        for i in range(testNorm.shape[0]):
            result = classify0(testNorm[i], trainNorm, trainLabels, kk)
            if result != testLabels[i]:
                errors += 1
        acc = 100.0 * (1.0 - errors / float(testNorm.shape[0]))
        accuracy.append(acc)

    plt.figure()
    plt.plot(range(1, 31), accuracy, marker='o')
    plt.xticks(range(1, 31))
    plt.xlabel('K')
    plt.ylabel('准确率(%)')
    plt.title('KNN K-值 与 准确率')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


# 字体设置：增加备用字体并关闭 Unicode minus，避免 SimHei 缺字形带来的警告
fm = FontManager()
mat_fonts = set(f.name for f in fm.ttflist)
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    datingClassTest()