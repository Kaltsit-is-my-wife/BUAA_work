from numpy import *
from matplotlib.font_manager import FontManager
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# TODO: 读取指定格式的数据文件，并将每一行的数据分为特征和标签
def file2matrix(filename):
    # TODO: 首先打开文件，统计行数以便初始化存储空间
    fr = open(filename)
    numberOfLines = len(fr.readlines())  # 通过读取所有行计算样本总数
    # TODO: 创建一个特征矩阵（每行一个样本，三列特征），以及对应的标签列表
    returnMat = zeros((numberOfLines, 3))  # 存储所有样本的特征数据
    classLabelVector = []  # 存储所有样本的类别标签
    fr = open(filename)
    index = 0
    # TODO: 对每一行数据，去除空白后按照分隔符分割，并分别存入矩阵和标签
    for line in fr.readlines():
        line = line.strip()  # 去掉行首行尾的空白字符，保证数据干净
        listFromLine = line.split('\t')  # 按照制表符分割字段，得到特征和标签
        returnMat[index, :] = listFromLine[0:3]  # 把前三个字段（特征）存入矩阵
        classLabelVector.append(int(listFromLine[-1]))  # 最后一个字段作为样本类别，转为整数
        index += 1
    # TODO: 返回处理完毕的特征矩阵和标签列表，便于后续建模使用
    return returnMat, classLabelVector

# TODO: 对输入的数据集进行归一化处理，让所有特征值映射到同一数值范围，统一一下便于距离计算
def autoNorm(dataSet):
    minVals = dataSet.min(0)  # 获取每个特征的最小值
    maxVals = dataSet.max(0)  # 获取每个特征的最大值
    ranges = maxVals - minVals  # 计算每个特征的取值范围
    normDataSet = zeros(shape(dataSet))  # 初始化归一化后数据的存储空间
    m = dataSet.shape[0]  # 样本数量
    # TODO: 将每个样本的特征值减去对应特征的最小值
    normDataSet = dataSet - tile(minVals, (m, 1))
    # TODO: 再将每个样本的特征值除以对应特征的范围，实现缩放到0~1区间
    normDataSet = normDataSet / tile(ranges, (m, 1))
    # TODO: 返回归一化后的数据、范围和最小值，这些参数也可用于逆归一化或其它处理
    return normDataSet, ranges, minVals

# TODO: 实现K近邻分类算法，对新输入样本inX进行类别预测
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # 获取训练集中样本的数量，用于距离计算
    # TODO: 复制输入样本，和所有训练样本分别相减，计算每个维度上的距离
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # 得到样本间每个特征的差值矩阵
    # TODO: 对差值矩阵各元素进行平方，方便后面算距离
    sqDiffMat = diffMat ** 2
    # TODO: 按行求和，将每个样本与输入样本的各维度差的平方累加
    sqDistances = sqDiffMat.sum(axis=1)
    # TODO: 对平方和开方，得到实际的距离
    distances = sqDistances ** 0.5
    # TODO: 对所有距离进行排序，返回距离最近的样本的索引
    sortedDistIndicies = distances.argsort()
    classCount = {}  # 初始化类别计数器，用于投票统计
    # TODO: 遍历距离最近的k个训练样本，统计每个类别出现次数
    for i in range(k):
        label = labels[sortedDistIndicies[i]]  # 取第i近的样本的类别
        classCount[label] = classCount.get(label, 0) + 1  # 累计该类别的票数
    # TODO: 找出票数最多的类别，作为最终的预测结果返回
    maxClassCount = max(classCount, key=classCount.get)
    return maxClassCount

# TODO: 用KNN算法测试不同K值下的分类准确率，并用折线图进行可视化
def datingClassTest():
    # TODO: 读取测试集和训练集数据，分别获得特征和标签
    testData, testLabels = file2matrix('knn_testData.txt')  # 测试数据
    trainData, trainLabels = file2matrix('knn_trainData.txt')  # 训练数据

    # TODO: 分别对测试集和训练集进行归一化，使特征值在统一范围便于距离计算
    testNorm, _r, _m = autoNorm(testData)  # 测试集归一化
    trainNorm, _rr, _mm = autoNorm(trainData)  # 训练集归一化
    errorCount = 0.0  # 初始化错误预测计数器
    accuracy = []  # 用于存储每个K值的正确率
    # TODO: 遍历不同的K值（从1到30），逐个测试并统计准确率
    for kk in range(1, 31):
        for i in range(testNorm.shape[0]):  # 对测试集每个样本进行预测
            result = classify0(testNorm[i], trainNorm, trainLabels, kk)  # 用当前K值做预测
            if (result != testLabels[i]):  # 如果预测类别与真实类别不一致
                errorCount += 1.0  # 错误计数器加一
        # TODO: 计算当前K值的分类准确率，百分比形式保存
        accuracy.append(100 - errorCount / float(testNorm.shape[0]) * 100)
        # TODO: 归零错误计数器，为下一个K值的统计做准备
        errorCount = 0.0

    # TODO: 利用matplotlib绘制K值与准确率的关系曲线，观察K对性能的影响
    plt.plot(range(1, 31), accuracy)
    # TODO: 设置横轴标签，表明横轴是K的取值范围
    plt.xlabel('K')
    # TODO: 设置纵轴标签，表明纵轴显示的是分类准确率百分比
    plt.ylabel('准确率(%)')
    # TODO: 展示图表，使结果可视化便于分析
    plt.show()


fm = FontManager()
mat_fonts = set(f.name for f in fm.ttflist)
plt.rcParams['font.sans-serif'] = ['SimHei']

# TODO: 主程序入口，运行准确率分析函数，输出KNN算法在不同K值下的表现
if __name__ == '__main__':
    datingClassTest()