import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import myTool as mT
import analyticalSolution as aS


class GradientDescent(object):

    def __init__(self, x, t, m, ln_lam, alpha, delta=10 ** (-6)):
        # 初始化行向量x,列向量T,多项式阶数m,超参数lambda,迭代步长alpha,精度delta
        self.x = x
        self.T = t
        self.m = m
        self.lam = np.exp(ln_lam)
        self.alpha = alpha
        self.delta = delta

    def cal_loss(self, w):
        # 求解当前多项式的系数w对应的拟合优度,用E_rms体现
        # w为列向量，其中每个元素为拟合所用的多项式的系数
        y = mT.get_predictive_y(self.x, w, self.m)
        return mT.cal_e_rms(y, self.T, w, self.lam)

    def cal_gradient(self, w):
        # 求代价函数的梯度
        # w为列向量，其中每个元素为拟合所用的多项式的系数
        matrix_x = mT.get_matrix_x(self.x, self.m)
        return matrix_x.T @ matrix_x @ w - matrix_x.T @ self.T + self.lam * w

    def solve(self, w0):
        # 迭代求解最优解w*
        # w0为列向量，其中每个元素为拟合所用的多项式的系数，迭代开始时的初始值
        w = w0
        gradient = self.cal_gradient(w0)

        times = 0  # 记录迭代次数

        times_list = [times]
        loss_list = [self.cal_loss(w0)]

        while not np.all(np.absolute(gradient) <= self.delta):
            times += 1
            w = w - self.alpha * gradient
            gradient = self.cal_gradient(w)
            times_list.append(times)
            loss_list.append(self.cal_loss(w))

        return w, np.array(times_list), np.array(loss_list)

    def solve_without_loss(self, w0):
        # 迭代求解最优解w*
        # w0为列向量，其中每个元素为拟合所用的多项式的系数，迭代开始时的初始值
        w = w0
        gradient = self.cal_gradient(w0)

        times = 0  # 记录迭代次数

        times_list = [times]

        while not np.all(np.absolute(gradient) <= self.delta):
            times += 1
            w = w - self.alpha * gradient
            gradient = self.cal_gradient(w)
            times_list.append(times)

        return w, np.array(times_list)


def different_alpha_e_rms(sigma, n_train, m, ln_lam, alpha_range):
    # 不同alpha取值下，E_rms随迭代次数的变化情况
    # sigma为数据噪声的标准差
    # n_train为训练集大小
    # m为拟合时的多项式的阶数
    # ln_lam为超参数lambda取对数后的值
    # alpha_range为超参数alpha的取值范围

    x_train, t_train = mT.generate_data_with_noise(sigma, n_train)

    fig, axes = plt.subplots()

    color_list = ['r', 'g', 'b']
    i = -1
    for alpha in alpha_range:
        i += 1
        color = color_list[i]
        # 通过梯度下降求解
        gradient_descent = GradientDescent(x_train, t_train.T, m, ln_lam, alpha)
        w0 = np.zeros(m + 1).T
        w, times_list, loss_list = gradient_descent.solve(w0)

        axes.plot(times_list, loss_list, color=color, label="$\\alpha = $" + str(alpha))

    title = "Iterative Results for Different alpha"
    props = {'title': title, 'xlabel': '$Times$', 'ylabel': '$E_{RMS}$'}
    axes.set(**props)
    axes.legend()
    axes.set_ylim(bottom=0)
    plt.show()


def gradient_descent_and_with_punishment(sigma, n_train, n_fit, m, ln_lam, alpha):
    # 对比梯度下降和有惩罚项的拟合曲线
    # sigma为数据噪声的标准差
    # n_train为训练集大小
    # n_fit为用来拟合的数据集大小
    # m为拟合时的多项式的阶数
    # ln_lam为超参数lambda取对数后的值
    # alpha为超参数alpha的取值

    x_train, t_train = mT.generate_data_with_noise(sigma, n_train)
    x_fit = np.linspace(0, 1, n_fit)
    fig, axes = plt.subplots()

    # 画训练集中各点
    # 默认情况下绘制线性回归拟合直线，用fit_reg = False将其删除
    sns.regplot(x=x_train, y=t_train, fit_reg=False, color="b", label="Training Set", ax=axes)
    # 画待拟合函数
    sns.lineplot(x=x_fit, y=np.sin(2 * np.pi * x_fit), color="g", label="$\\sin(2\\pi x)$", ax=axes)
    # 画多项式拟合曲线（带惩罚项的解析解）
    analytical_solution = aS.AnalyticalSolution(mT.get_matrix_x(x_train, m), t_train.T)
    w1 = analytical_solution.with_punishment(np.exp(ln_lam))
    sns.lineplot(x=x_fit, y=mT.get_predictive_y(x_fit, w1, m), color="r", label="Analytical With Punishment",
                 ax=axes)
    # 画多项式拟合曲线（梯度下降）
    gradient_descent = GradientDescent(x_train, t_train.T, m, ln_lam, alpha)
    w0 = np.zeros(m + 1).T
    w2, times_list = gradient_descent.solve_without_loss(w0)
    sns.lineplot(x=x_fit, y=mT.get_predictive_y(x_fit, w2, m), color="purple", label="Gradient Descent", ax=axes)

    axes.set_title('M = ' + str(m) + ', N_train = ' + str(n_train), fontsize=16)
    plt.show()


def different_n_train(sigma, n_train_range, n_fit, m, ln_lam, alpha, layout):
    # 用不同训练集大小分别训练，以拟合正弦函数（梯度下降）
    # sigma为数据噪声的标准差
    # n_train_range为不同的用来训练的数据集大小
    # n_fit为用来拟合的数据集大小
    # m为拟合时的多项式的阶数
    # ln_lam为超参数lambda取对数后的值
    # alpha为超参数alpha的取值
    # layout为画图时排版格式（行数和列数）

    fig, axes = plt.subplots(*layout)
    # 对每个n_train分别进行训练后拟合并画图
    for i in range(len(n_train_range)):
        n_train = n_train_range[i]
        x_train, t_train = mT.generate_data_with_noise(sigma, n_train)
        x_fit = np.linspace(0, 1, n_fit)
        # 画图操作
        # 确定画图位置
        ax_target = axes[i // layout[1]][i % layout[1]]
        # 画训练集中各点
        # 默认情况下绘制线性回归拟合直线，用fit_reg = False将其删除
        sns.regplot(x=x_train, y=t_train, fit_reg=False, color="b", label="training set", ax=ax_target)
        # 画待拟合函数
        sns.lineplot(x=x_fit, y=np.sin(2 * np.pi * x_fit), color="g", label="$\\sin(2\\pi x)$", ax=ax_target)
        # 画多项式拟合曲线
        gradient_descent = GradientDescent(x_train, t_train.T, m, ln_lam, alpha)
        w0 = np.zeros(m + 1).T
        w, times_list = gradient_descent.solve_without_loss(w0)
        sns.lineplot(x=x_fit, y=mT.get_predictive_y(x_fit, w, m), color="r", label="fit result", ax=ax_target)
        # 设置图名
        title = "M = " + str(m) + ", N_train = " + str(n_train) + ", Times = " + str(times_list[-1])
        props = {'title': title}
        ax_target.set(**props)

    plt.show()


def different_m(sigma, n_train, n_fit, m_range, ln_lam, alpha, layout):
    # 用不同多项式阶数分别拟合正弦函数（梯度下降）
    # sigma为数据噪声的标准差
    # n_train为训练集大小
    # n_fit为用来拟合的数据集大小
    # m_range为拟合时的多项式的不同阶数
    # ln_lam为超参数lambda取对数后的值
    # alpha为超参数alpha的取值
    # layout为画图时排版格式（行数和列数）

    x_train, t_train = mT.generate_data_with_noise(sigma, n_train)
    fig, axes = plt.subplots(*layout)
    # 对每个n_train分别进行训练后拟合并画图
    for i in range(len(m_range)):
        order = m_range[i]
        x_fit = np.linspace(0, 1, n_fit)
        # 画图操作
        # 确定画图位置
        ax_target = axes[i // layout[1]][i % layout[1]]
        # 画训练集中各点
        # 默认情况下绘制线性回归拟合直线，用fit_reg = False将其删除
        sns.regplot(x=x_train, y=t_train, fit_reg=False, color="b", label="training set", ax=ax_target)
        # 画待拟合函数
        sns.lineplot(x=x_fit, y=np.sin(2 * np.pi * x_fit), color="g", label="$\\sin(2\\pi x)$", ax=ax_target)
        # 画多项式拟合曲线
        gradient_descent = GradientDescent(x_train, t_train.T, order, ln_lam, alpha)
        w0 = np.zeros(order + 1).T
        w, times_list = gradient_descent.solve_without_loss(w0)
        sns.lineplot(x=x_fit, y=mT.get_predictive_y(x_fit, w, order), color="r", label="fit result", ax=ax_target)
        # 设置图名
        title = "M = " + str(order) + ", N_train = " + str(n_train) + ", Times = " + str(times_list[-1])
        props = {'title': title}
        ax_target.set(**props)

    plt.show()
