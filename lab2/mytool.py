import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import shuffle


def generate_data(mu_0, mu_1, sigma, n_0, n_1, train_rate, cov=0.0):
    """
    生成两个类别的数据（高斯分布），默认生成的是满足朴素贝叶斯的数据（cov=0.0），若要生成不满足朴素贝叶斯的数据，则需传入另外的cov值
    :param mu_0: 类别0中数据的均值（默认各维度均值相等）
    :param mu_1: 类别1中数据的均值（默认各维度均值相等）
    :param sigma: 两个类别中数据的标准差（默认各维度标准差相等）
    :param n_0: 类别0中数据量
    :param n_1: 类别1中数据量
    :param train_rate: 每个类别中训练集数据占总数据的比例（剩下部分为测试集）
    :param cov: 数据两个维度(x和y)的协方差，即cov(x,y)和cov(y,x)，默认情况下为零，独立，满足朴素贝叶斯
    :return: x_train,y_train,x_test,y_test: 生成的训练集与测试集
    """
    # 分别生成两个类别的数据
    data_1 = np.random.multivariate_normal((mu_0, mu_0), [[sigma, cov], [cov, sigma]], n_0)
    data_2 = np.random.multivariate_normal((mu_1, mu_1), [[sigma, cov], [cov, sigma]], n_1)
    # 将训练集与测试集划分开
    data_sep_1 = int(train_rate * n_0)
    data_sep_2 = int(train_rate * n_1)
    data_1_train, data_1_test = data_1[:data_sep_1], data_1[data_sep_1:]
    data_2_train, data_2_test = data_2[:data_sep_2], data_2[data_sep_2:]
    # 将两个类别的数据都画到图上
    plt.scatter(data_1_train.T[0], data_1_train.T[1], color='g')
    plt.xlabel('$X_{1}$')
    plt.scatter(data_2_train.T[0], data_2_train.T[1], color='r')
    plt.ylabel('$X_{2}$')
    # 将两个类别中的训练集数据与测试集数据分别合在一起
    x_train = np.concatenate([data_1_train, data_2_train])
    y_train = np.concatenate([np.zeros(data_1_train.shape[0]), np.ones(data_2_train.shape[0])])
    x_test = np.concatenate([data_1_test, data_2_test])
    y_test = np.concatenate([np.zeros(data_1_test.shape[0]), np.ones(data_2_test.shape[0])])
    # 将训练集中数据打乱
    x_train, y_train = shuffle(x_train, y_train)
    return x_train, y_train, x_test, y_test


def load_data(path, train_rate):
    """
    使用从UCI获取的数据集
    :param path: 数据集所在路径
    :param train_rate: 每个类别中训练集数据占总数据的比例（剩下部分为测试集）
    :return: x_train,y_train,x_test,y_test: 获取的训练集与测试集
    """
    data = []
    file = open(path, encoding='utf-8')
    for line in file:
        data.append(line.strip('\n').split(sep=','))
    # 将数据分为0类和1类
    all_data = np.array(data)
    all_data = np.where(all_data == 'Iris-setosa', 0, all_data)
    all_data = np.where(all_data == 'Iris-versicolor', 1, all_data)
    all_data = np.where(all_data == 'Iris-virginica', 1, all_data)
    # 将数据与类别标签分开
    x = all_data[:, :len(all_data[0]) - 1]
    y = all_data[:, -1]
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=int)
    # 把数据集进行归一化处理
    x = (x-np.mean(x, axis=0))/(np.std(x, axis=0))
    # 将数据集打乱并分为训练集和测试集
    data_sep = int(train_rate * len(all_data))
    x, y = shuffle(x, y)
    x_train = x[:data_sep, :]
    x_test = x[data_sep:, :]
    y_train = y[:data_sep]
    y_test = y[data_sep:]
    return x_train, y_train, x_test, y_test


def draw_loss_line(t, loss_list):
    """
    画出loss随迭代次数的变化情况
    :param t: 横坐标，迭代次数
    :param loss_list: 纵坐标，代价函数值
    """
    fig, axes = plt.subplots()
    axes.plot(t, loss_list, 'b')
    # 设置图名
    title = "Loss for Different Iterative Times"
    props = {'title': title, 'xlabel': 'times', 'ylabel': 'loss'}
    axes.set(**props)
    plt.show()
