import numpy as np
from matplotlib import pyplot as plt


class LogisticRegression(object):

    def __init__(self, dimension):
        """
        初始化逻辑回归分类器的参数
        :param dimension: 数据的维度
        """
        self.w = np.random.randn(dimension)
        self.b = np.random.randn(1)

    def sigmoid(self, x):
        """
        将数据值映射到sigmoid函数上
        :param x: 数据
        :return: 数据在当前参数取值下，映射到的sigmoid函数值
        """
        return 1 / (1 + np.e**(-(x.dot(self.w) + self.b)))

    def predict(self, x):
        """
        预测输入的数据属于哪一个类（二分类）
        :param x: 数据集
        :return: 数据所属类别，0为0类，1为1类
        """
        result = self.sigmoid(x)
        result = np.where(result >= 0.5, 1, 0)
        return result

    def cal_loss(self, x, y):
        """
        计算代价函数
        :param x: 数据集
        :param y: 各个数据的分类情况
        :return: 该组数据的代价函数值
        """
        return -sum(y * (x.dot(self.w) + self.b) - np.log(1 + np.e**(x.dot(self.w) + self.b)))

    def cal_gradient(self, x, y, lam):
        """
        计算w和b的梯度
        :param x: 数据集
        :param y: 各个数据的分类情况
        :param lam: 惩罚项比重
        :return: w和b的梯度
        """
        sig = self.sigmoid(x)
        w_gradient = -((y - sig).dot(x) + lam * self.w)
        b_gradient = -(np.sum(y - sig) + lam * self.b)
        return w_gradient, b_gradient

    def solve(self, x, y, eta, times, lam=0.0):
        """
        梯度下降法求解w和b
        :param x: 数据集
        :param y: 各个数据的分类情况
        :param eta: 迭代步长
        :param times: 迭代次数上限
        :param lam: 惩罚项比重
        :return: loss_list, range(t + 1): 代价函数与迭代次数
        """
        loss_list = [self.cal_loss(x, y)]
        w_gradient, b_gradient = self.cal_gradient(x, y, lam)
        t = 0
        while not (np.all(np.absolute(w_gradient) <= 1e-5) and np.all(np.absolute(b_gradient) <= 1e-5)):
            if t >= times:
                break
            self.w -= eta * w_gradient
            self.b -= eta * b_gradient
            loss_list.append(self.cal_loss(x, y))
            w_gradient, b_gradient = self.cal_gradient(x, y, lam)
            t += 1
        print("final loss of the training set:", self.cal_loss(x, y))
        print("times of iteration:", t)
        return loss_list, range(t + 1)

    def accuracy(self, x, y):
        """
        :param x: 数据集
        :param y: 各个数据的分类情况
        :return: 逻辑回归分类器在该组数据上的正确率
        """
        y_pre = self.predict(x)
        count = 0
        for i in range(len(y)):
            if y[i] == y_pre[i]:
                count += 1
        return count / len(y)

    def draw_border(self, low, high):
        """
        画出逻辑回归分类器的决策边界（仅支持二维图）
        :param low: 横坐标最小值
        :param high: 横坐标最大值
        """
        if len(self.w) != 2:
            print("only 2D pictures are supported")
            return
        x = np.linspace(low, high, 1000)
        y = (-self.b - self.w[0] * x) / self.w[1]
        plt.plot(x, y)
        plt.show()
