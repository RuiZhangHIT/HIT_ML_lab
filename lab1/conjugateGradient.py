import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import myTool as mT
import gradientDescent as gD


class ConjugateGradient(object):
    def __init__(self, x, t, m, ln_lam, delta=10 ** (-4)):
        # 初始化行向量X,列向量T,多项式阶数m,超参数lambda,精度delta
        self.x = x
        self.T = t
        self.order = m
        self.lam = np.exp(ln_lam)
        self.delta = delta

    def cal_a(self):
        # 求解代价函数E(w)写成二次型后的矩阵A
        matrix_x = mT.get_matrix_x(self.x, self.order)
        return matrix_x.T @ matrix_x + self.lam * np.identity(len(matrix_x.T))

    def cal_b(self):
        # 求解代价函数E(w)写成二次型后的矩阵b
        matrix_x = mT.get_matrix_x(self.x, self.order)
        return matrix_x.T @ self.T

    def cal_loss(self, w):
        # 求解当前多项式的系数w对应的拟合优度,用E_rms体现
        # w为列向量，其中每个元素为拟合所用的多项式的系数
        y = mT.get_predictive_y(self.x, w, self.order)
        return mT.cal_e_rms(y, self.T, w, self.lam)

    def solve(self, a, b, w0):
        # 迭代求解最优解w*
        # a为代价函数E(w)写成二次型后的矩阵A
        # b为代价函数E(w)写成二次型后的矩阵b
        # w0为列向量，其中每个元素为拟合所用的多项式的系数，迭代开始时的初始值

        w = w0
        times = 0
        times_list = [times]  # 记录迭代次数
        loss_list = [self.cal_loss(w)]  # 记录拟合优度
        r = [b - a @ w]  # 记录残差向量
        p = [b - a @ w]  # 记录搜索方向

        while True:
            times += 1
            times_list.append(times)

            if times == 1:  # 第一次迭代搜索方向由残差向量确定
                p.append(r[0])
            else:  # 后续迭代搜索方向需计算确定
                p.append(
                    r[times - 1] + (r[times - 1].T @ r[times - 1]) / (r[times - 2].T @ r[times - 2]) * p[times - 1])
            alpha = (r[times - 1].T @ r[times - 1]) / (p[times].T @ a @ p[times])
            w = w + alpha * p[times]
            loss_list.append(self.cal_loss(w))
            r.append(r[times - 1] - alpha * a @ p[times])

            if np.all(np.absolute(r[times]) <= self.delta):
                break

        return w, np.array(times_list), np.array(loss_list)


def gradient_descent_and_conjugate_gradient(sigma, n_train, n_fit, m, ln_lam, alpha):
    # 对比梯度下降和共轭梯度的拟合曲线
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
    # 画多项式拟合曲线（共轭梯度）
    conjugate_gradient = ConjugateGradient(x_train, t_train.T, m, ln_lam)
    a = conjugate_gradient.cal_a()
    b = conjugate_gradient.cal_b()
    w0 = np.zeros(m + 1).T
    w1, times_list, loss_list = conjugate_gradient.solve(a, b, w0)
    sns.lineplot(x=x_fit, y=mT.get_predictive_y(x_fit, w1, m), color="r", label="Conjugate Gradient", ax=axes)
    # 画多项式拟合曲线（梯度下降）
    gradient_descent = gD.GradientDescent(x_train, t_train.T, m, ln_lam, alpha)
    w0 = np.zeros(m + 1).T
    w2, times_list = gradient_descent.solve_without_loss(w0)
    sns.lineplot(x=x_fit, y=mT.get_predictive_y(x_fit, w2, m), color="purple", label="Gradient Descent", ax=axes)
    axes.set_title('M = ' + str(m) + ', N_train = ' + str(n_train), fontsize=16)

    plt.show()


def different_n_train(sigma, n_train_range, n_fit, m, ln_lam, layout):
    # 用不同训练集大小分别训练，以拟合正弦函数（共轭梯度）
    # sigma为数据噪声的标准差
    # n_train_range为不同的用来训练的数据集大小
    # n_fit为用来拟合的数据集大小
    # m为拟合时的多项式的阶数
    # ln_lam为超参数lambda取对数后的值
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
        conjugate_gradient = ConjugateGradient(x_train, t_train.T, m, ln_lam)
        a = conjugate_gradient.cal_a()
        b = conjugate_gradient.cal_b()
        w0 = np.zeros(m + 1).T
        w, times_list, loss_list = conjugate_gradient.solve(a, b, w0)
        sns.lineplot(x=x_fit, y=mT.get_predictive_y(x_fit, w, m), color="r", label="fit result", ax=ax_target)
        # 设置图名
        title = "M = " + str(m) + ", N_train = " + str(n_train) + ", Times = " + str(times_list[-1])
        props = {'title': title}
        ax_target.set(**props)

    plt.show()


def different_m(sigma, n_train, n_fit, m_range, ln_lam, layout):
    # 用不同多项式阶数分别拟合正弦函数（共轭梯度）
    # sigma为数据噪声的标准差
    # n_train为训练集大小
    # n_fit为用来拟合的数据集大小
    # m_range为拟合时的多项式的不同阶数
    # ln_lam为超参数lambda取对数后的值
    # layout为画图时排版格式（行数和列数）

    x_train, t_train = mT.generate_data_with_noise(sigma, n_train)
    fig, axes = plt.subplots(*layout)
    # 对每个n_train分别进行训练后拟合并画图
    for i in range(len(m_range)):
        m = m_range[i]
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
        conjugate_gradient = ConjugateGradient(x_train, t_train.T, m, ln_lam)
        a = conjugate_gradient.cal_a()
        b = conjugate_gradient.cal_b()
        w0 = np.zeros(m + 1).T
        w, times_list, loss_list = conjugate_gradient.solve(a, b, w0)
        sns.lineplot(x=x_fit, y=mT.get_predictive_y(x_fit, w, m), color="r", label="fit result", ax=ax_target)
        # 设置图名
        title = "M = " + str(m) + ", N_train = " + str(n_train) + ", Times = " + str(times_list[-1])
        props = {'title': title}
        ax_target.set(**props)

    plt.show()
