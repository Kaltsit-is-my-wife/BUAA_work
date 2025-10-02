import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# TODO:从CSV文件中加载数据
data = pd.read_csv('data/weights_heights.csv')

# TODO:打印前5行数据
print(data.head())

# TODO:设置数据的点图，x轴为Weight，y轴为Height
plt.scatter(data['Weight'], data['Height'])
plt.xlabel('Weight (lbs)')
plt.ylabel('Height (Inch)')
plt.grid()
plt.show()

# TODO:把整列数据取出来，分别赋值给X和y
X, y = data['Weight'].values, data['Height'].values

# TODO:调用train_test_split划分训练集和验证集，比例为7:3，随机种子设为17，其中的参数X,y按顺序处理后按相同的顺序传回
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=17)

# TODO:创建一个StandardScaler对象，用他的fit_transform方法分别处理X_train和X_valid
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape([-1, 1]))
X_valid_scaled = scaler.fit_transform(X_valid.reshape([-1, 1]))


class SGDRegressor():

    # TODO:初始化函数，学习率learnRate设置为0.001，迭代次数epochs=3
    def __init__(self, lr=1e-3, epochs=3):
        self.lr = lr
        self.epochs = epochs
        self.mse = []
        self.weights = []

    # TODO:定义fit函数，输入参数为X和y，就是之前经过预处理的训练集。
    # TODO:在函数中，先给X加上偏置项，然后初始化权重w为0向量，长度为X的列数。
    # TODO:按照传入的迭代次数epochs进行循环，每次循环中，对训练集中的每一个样本进行如下操作：
    # TODO:先复制当前的权重w到new_w中，然后更新new_w的每一个分量，更新公式为：
    # TODO:new_w[0] += learnRate * (y[i] - w.dot(X[i, :])),然后每轮都更新一个项
    # TODO:调用mse记录当前的均方误差，最后从所有记录的均方误差中选出最小的对应的权重作为最终的权重self.w
    def fit(self, X, y):
        X = np.hstack([np.ones([X.shape[0], 1]), X])
        w = np.zeros(X.shape[1])
        for it in tqdm(range(self.epochs)):
            for i in range(X.shape[0]):
                new_w = w.copy()
                new_w[0] += self.lr * (y[i] - w.dot(X[i, :]))
                for j in range(1, X.shape[1]):
                    new_w[j] += self.lr * (y[i] - w.dot(X[i, :])) * X[i, j]
                w = new_w.copy()
                self.weights.append(w)
                self.mse.append(mean_squared_error(y, X.dot(w)))
        self.w = self.weights[np.argmin(self.mse)]
        return self

    # TODO:定义predict函数，输入参数为X，先给X加上偏置项，然后返回X和权重w的点积
    def predict(self, X):
        X = np.hstack([np.ones([X.shape[0], 1]), X])
        return X.dot(self.w)


# TODO:创建SGDRegressor对象sgd_reg，调用fit函数传入训练集X_train_scaled和y_train进行训练
sgd_reg = SGDRegressor()
sgd_reg.fit(X_train_scaled, y_train)

# TODO:绘制误差曲线，横轴为更新次数，纵轴为均方误差
plt.plot(range(len(sgd_reg.mse)), sgd_reg.mse)
plt.xlabel('#updates')
plt.ylabel('MSE')
plt.show()

# TODO:打印最小均方误差和对应的权重
print("min_mse=", np.min(sgd_reg.mse), "w=", sgd_reg.w)

# TODO:绘制权重变化曲线，横轴为更新次数，纵轴为权重值
plt.subplot(121)
plt.plot(range(len(sgd_reg.weights)), [w[0] for w in sgd_reg.weights])
plt.xlabel('#updates')
plt.ylabel('Weight')
plt.subplot(122)
plt.plot(range(len(sgd_reg.weights)), [w[1] for w in sgd_reg.weights])
plt.xlabel('#updates')
plt.ylabel('Coefficient')
plt.show()

# TODO:打印验证集的均方误差
sgd_valid_mse = mean_squared_error(y_valid, sgd_reg.predict(X_valid_scaled))
print("sgd_valid_mse=", sgd_valid_mse)
