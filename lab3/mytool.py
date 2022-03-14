import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

plt.style.use('seaborn')


def generate_data(mus, sigmas, nums, dim, show=True):
    """
    生成num个高斯分布的数据
    :param mus: 各类高斯分布的均值
    :param sigmas: 各类高斯分布的标准差
    :param nums: 高斯分布的个数
    :param dim: 数据的维度
    :param show: 是否展示最终生成数据集和类别情况图
    :return: 生成的数据集
    """
    datasets = []
    for mu, sigma, num in zip(mus, sigmas, nums):
        # 临时存储一个高斯分布的数据集
        dataset = np.random.randn(num, dim)
        for i, v in enumerate(sigma):
            dataset[:, i] *= sigma[i]
        for i, m in enumerate(mu):
            dataset[:, i] += mu[i]
        datasets.append(dataset)
    if show:
        for dataset in datasets:
            # 画出每一个数据集
            plt.scatter(dataset.T[0], dataset.T[1])
        plt.show()
    datasets = np.concatenate(datasets)
    return np.array(datasets, dtype=float)


def load_data(path):
    """
    使用从UCI获取的数据集
    :param path: 数据集所在路径
    :return: 获取的数据集与标签，各类的均值与标准差
    """
    data = []
    file = open(path, encoding='utf-8')
    for line in file:
        data.append(line.strip('\n').split(sep=','))
    # 将数据分为0类、1类和2类
    all_data = np.array(data)
    all_data = np.where(all_data == 'Iris-setosa', 0, all_data)
    all_data = np.where(all_data == 'Iris-versicolor', 1, all_data)
    all_data = np.where(all_data == 'Iris-virginica', 2, all_data)
    # 将数据与类别标签分开
    x = all_data[:, :len(all_data[0]) - 1]
    y = all_data[:, -1]
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=int)
    # 把数据集进行归一化处理
    x = (x - np.mean(x, axis=0))/(np.std(x, axis=0))
    # 将数据集打乱
    x, y = shuffle(x, y)
    return x, y


def color_data(data):
    """
    将数据集分类情况画图展示出
    :param data: 数据集及其标签
    :return: 分类情况图
    """
    if data.shape[1] != 3:
        print("only 2D pictures are supported")
        return
    # 计数，分离数据与类标签
    num = data.shape[0]
    label = data[:, -1]
    x = data[:, :-1]

    color = ['r', 'orange', 'yellow', 'g', 'b', 'purple', 'pink']
    # 按类标签选色画图
    for i in range(num):
        plt.scatter(x[i, :].T[0], x[i, :].T[1], color=color[int(label[i])])
    plt.show()


def cal_accuracy(label_pre, label_real):
    """
    计算分类的准确度
    :param label_pre: 预测的分类标签
    :param label_real: 实际的分类标签
    :return: 分类准确度
    """
    count = 0
    num = len(label_real)
    for i in range(num):
        # 无监督学习需要考虑标签名不对应的情况，以下为试验后的匹配情况
        if label_real[i] == 0:
            if label_pre[i] == 1:
                count += 1
        elif label_real[i] == 1:
            if label_pre[i] == 2:
                count += 1
        else:
            if label_pre[i] == 0:
                count += 1
    return count / num
