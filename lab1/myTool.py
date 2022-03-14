import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def generate_data_with_noise(sigma, n):
    # 生成带（高斯）噪声的数据
    # sigma为噪声的标准差，默认噪声的均值为0
    # n为需要生成的数据集的大小

    x = np.linspace(0, 1, n)
    noise = np.random.normal(0, sigma, n)
    t = np.sin(2 * np.pi * x) + noise
    return x, t


def different_sigma(sigma_range, n_train, n_exact, layout):
    # 用不同标准差分别生成带（高斯）噪声的数据
    # sigma_range为不同的标准差
    # n_train为用来训练的数据集大小
    # n_exact为从待拟合函数（正弦函数）上取的数据集大小
    # layout为画图时排版格式（行数和列数）

    fig, axes = plt.subplots(*layout)
    # 对每个sigma分别生成数据并画图
    for i in range(len(sigma_range)):
        sigma = sigma_range[i]
        x_exact = np.linspace(0, 1, n_exact)
        x_train, t_train = generate_data_with_noise(sigma, n_train)
        # 画图操作
        # 确定画图位置
        ax_target = axes[i // layout[1]][i % layout[1]]
        # 画训练集中各点
        # 默认情况下绘制线性回归拟合直线，用fit_reg = False将其删除
        sns.regplot(x=x_train, y=t_train, fit_reg=False, color="b", label="training set", ax=ax_target)
        # 画待拟合函数
        sns.lineplot(x=x_exact, y=np.sin(2 * np.pi * x_exact), color="g", label="$\\sin(2\\pi x)$", ax=ax_target)
        # 设置图名
        title = 'DataSet with sigma: ' + str(sigma)
        props = {'title': title}
        ax_target.set(**props)

    plt.show()


def get_matrix_x(x, m):
    # 用给定行向量生成公式推导中的矩阵X
    # x为行向量，其中每个元素为待预测点的横坐标
    # m为拟合所用的多项式的阶数
    # 例如：
    #     输入：[x y z], m
    #     输出：[[1 x x^2 ... x^m]
    #           [1 y y^2 ... y^m]
    #           [1 z z^2 ... z^m]]

    matrix = np.ones((len(x), m + 1))
    for i in range(0, len(x)):
        for j in range(1, m + 1):
            matrix[i][j] = matrix[i][j - 1] * x[i]
    return matrix


def get_predictive_y(x, w, m):
    # 用推导公式中矩阵X和列向量w预测点的纵坐标列向量Y
    # x为行向量，其中每个元素为待预测点的横坐标
    # w为列向量，其中每个元素为拟合所用的多项式的系数
    # m为拟合所用的多项式的阶数

    matrix_x = get_matrix_x(x, m)
    return matrix_x @ w


def cal_e_rms(y, t, w, lam):
    # 计算方根均值，进行拟合优度评价
    # y为预测值
    # t为真实值
    # w为列向量，其中每个元素为拟合所用的多项式的系数
    # lam为公式推导中的超参数

    return np.sqrt(np.mean(np.square(y - t)) + lam * np.sqrt(w.T @ w) / len(t))
