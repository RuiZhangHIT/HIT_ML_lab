from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import myTool as mT


class AnalyticalSolution(object):

    def __init__(self, x, t):
        # 初始化X和T
        # x为公式推导中的矩阵，t为公式推导中的列向量
        self.X = x
        self.T = t

    def no_punishment(self):
        # 不带惩罚项的解析解求法
        return np.linalg.pinv(self.X) @ self.T

    def with_punishment(self, lam):
        # 带惩罚项的解析解求法
        # lam为公式推导中的超参数lambda
        return np.linalg.pinv(self.X.T @ self.X + lam * np.identity(len(self.X.T))) @ self.X.T @ self.T


def different_m(sigma, n_train, n_fit, m_range, layout):
    # 用不同阶数分别拟合正弦函数（无惩罚项）
    # sigma为数据噪声的标准差
    # n_train为用来训练的数据集大小
    # n_fit为用来拟合的数据集大小
    # m_range为不同的阶数
    # layout为画图时排版格式（行数和列数）

    fig, axes = plt.subplots(*layout)

    x_train, t_train = mT.generate_data_with_noise(sigma, n_train)
    x_fit = np.linspace(0, 1, n_fit)
    # 对每个order分别进行拟合并画图
    for i in range(len(m_range)):
        m = m_range[i]
        # 画图操作
        # 确定画图位置
        ax_target = axes[i // layout[1]][i % layout[1]]
        # 画训练集中各点
        # 默认情况下绘制线性回归拟合直线，用fit_reg = False将其删除
        sns.regplot(x=x_train, y=t_train, fit_reg=False, color="b", label="training set", ax=ax_target)
        # 画待拟合函数
        sns.lineplot(x=x_fit, y=np.sin(2 * np.pi * x_fit), color="g", label="$\\sin(2\\pi x)$", ax=ax_target)
        # 画多项式拟合曲线
        analytical_solution = AnalyticalSolution(mT.get_matrix_x(x_train, m), t_train.T)
        sns.lineplot(x=x_fit, y=mT.get_predictive_y(x_fit, analytical_solution.no_punishment(), m), color="r",
                     label="fit result", ax=ax_target)
        # 设置图名
        title = "M = " + str(m) + ", N_train = " + str(n_train)
        props = {'title': title}
        ax_target.set(**props)

    plt.show()


def different_n_train(sigma, n_train_range, n_fit, m, layout):
    # 用不同训练集大小分别训练，以拟合正弦函数（无惩罚项）
    # sigma为数据噪声的标准差
    # n_train_range为不同的用来训练的数据集大小
    # n_fit为用来拟合的数据集大小
    # m为拟合时的多项式的阶数
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
        analytical_solution = AnalyticalSolution(mT.get_matrix_x(x_train, m), t_train.T)
        sns.lineplot(x=x_fit, y=mT.get_predictive_y(x_fit, analytical_solution.no_punishment(), m), color="r",
                     label="fit result", ax=ax_target)
        # 设置图名
        title = "M = " + str(m) + ", N_train = " + str(n_train)
        props = {'title': title}
        ax_target.set(**props)

    plt.show()


def different_lam_e_rms(sigma, n_train, n_test, ln_lam_range):
    # 训练集E_rms与测试集E_rms随超参数lambda的变化情况
    # sigma为数据噪声的标准差
    # n_train为训练集大小
    # n_test为测试集大小
    # ln_lam_range为超参数lambda取对数后的取值范围

    # 生成训练集
    x_train, t_train = mT.generate_data_with_noise(sigma, n_train)
    # 生成测试集
    x_test, t_test = mT.generate_data_with_noise(sigma, n_test)
    # 存放训练集拟合优度
    e_rms_train_list = []
    # 存放测试集拟合优度
    e_rms_test_list = []

    e_rms_min = float('inf')
    best_ln_lam = 0
    for ln_lam in ln_lam_range:
        # 通过训练集求解得到w
        analytical_solution = AnalyticalSolution(mT.get_matrix_x(x_train, 9), t_train.T)
        w = analytical_solution.with_punishment(np.exp(ln_lam))
        # 按w对训练集进行结果预测，计算拟合优度并存入列表
        y_train = mT.get_predictive_y(x_train, w, 9)
        e_rms_train_list.append(mT.cal_e_rms(y_train, t_train.T, w, np.exp(ln_lam)))
        # 按w对测试集进行结果预测，计算拟合优度并存入列表
        y_test = mT.get_predictive_y(x_test, w, 9)
        e_rms = mT.cal_e_rms(y_test, t_test.T, w, np.exp(ln_lam))
        e_rms_test_list.append(e_rms)
        # 记录当前最优拟合情况
        if e_rms < e_rms_min:
            e_rms_min = e_rms
            best_ln_lam = ln_lam
    # 画图操作
    fig, axes = plt.subplots()
    # 训练集E_rms随超参数lambda的对数的变化情况
    axes.plot(ln_lam_range, e_rms_train_list, 'b-o', label="Training Set")
    # 测试集E_rms随超参数lambda的对数的变化情况
    axes.plot(ln_lam_range, e_rms_test_list, 'r-o', label="Test Set")
    # 设置图名
    title = "$E_{RMS}$ for Different ln(lambda), best_ln(lambda) : " + str(best_ln_lam)
    props = {'title': title, 'xlabel': '$ln(lambda)$', 'ylabel': '$E_{RMS}$'}
    axes.set(**props)
    axes.legend()
    axes.set_ylim(bottom=0)
    plt.show()


def find_lam(sigma, n_train, n_test, ln_lam_range, times):
    # 通过多次实验寻找使拟合最优的三个超参数取值的对数
    # sigma为噪声的标准差，默认噪声的均值为0
    # n_train为训练集大小
    # n_test为测试集大小
    # ln_lam_range为超参数lambda取对数后的取值范围
    # times为实验重复次数

    best_ln_lam_list = []
    for i in range(times):
        # 生成训练集
        x_train, t_train = mT.generate_data_with_noise(sigma, n_train)
        # 生成测试集
        x_test, t_test = mT.generate_data_with_noise(sigma, n_test)

        e_rms_min = float('inf')
        best_ln_lam = 0
        for ln_lam in ln_lam_range:
            # 通过训练集求解得到w
            analytical_solution = AnalyticalSolution(mT.get_matrix_x(x_train, 9), t_train)
            w = analytical_solution.with_punishment(np.exp(ln_lam))
            # 对测试集进行结果预测，计算拟合优度并记录下当前最优情况
            y_test = mT.get_predictive_y(x_test, w, 9)
            e_rms = mT.cal_e_rms(y_test, t_test, w, np.exp(ln_lam))
            if e_rms < e_rms_min:
                e_rms_min = e_rms
                best_ln_lam = ln_lam

        best_ln_lam_list.append(best_ln_lam)
    # 使拟合最优的三个超参数取值的对数及其被判为最优的实验次数
    best_ln_lams = Counter(best_ln_lam_list).most_common(3)
    print(best_ln_lams)


def with_and_no_punishment(sigma, n_train, n_fit, ln_lam):
    # 对比有惩罚项和无惩罚项的拟合曲线
    # sigma为数据噪声的标准差
    # n_train为训练集大小
    # n_fit为用来拟合的数据集大小
    # ln_lam为超参数lambda取对数后的值

    x_train, t_train = mT.generate_data_with_noise(sigma, n_train)
    x_fit = np.linspace(0, 1, n_fit)
    layout = (1, 2)
    fig, axes = plt.subplots(*layout)

    # 无惩罚项的图
    # 画训练集中各点
    sns.regplot(x=x_train, y=t_train, fit_reg=False, color="b", label="Training Set", ax=axes[0])
    # 画待拟合函数
    sns.lineplot(x=x_fit, y=np.sin(2 * np.pi * x_fit), color="g", label="$\\sin(2\\pi x)$", ax=axes[0])
    # 画多项式拟合曲线
    analytical_solution = AnalyticalSolution(mT.get_matrix_x(x_train, 9), t_train.T)
    sns.lineplot(x=x_fit, y=mT.get_predictive_y(x_fit, analytical_solution.no_punishment(), 9), color="r",
                 label="No Punishment", ax=axes[0])
    # 设置图名
    axes[0].set_title('Without Punishment', fontsize=16)

    # 带惩罚项的图
    # 画训练集中各点
    sns.regplot(x=x_train, y=t_train, fit_reg=False, color="b", label="Training Set", ax=axes[1])
    # 画待拟合函数
    sns.lineplot(x=x_fit, y=np.sin(2 * np.pi * x_fit), color="g", label="$\\sin(2\\pi x)$", ax=axes[1])
    # 画多项式拟合曲线
    analytical_solution = AnalyticalSolution(mT.get_matrix_x(x_train, 9), t_train.T)
    sns.lineplot(x=x_fit, y=mT.get_predictive_y(x_fit, analytical_solution.with_punishment(np.exp(ln_lam)), 9),
                 color="r", label="With Punishment", ax=axes[1])
    # 设置图名
    axes[1].set_title('With Punishment, ' + '$ln(\\lambda) = $' + str(ln_lam), fontsize=16)

    plt.show()


def different_m_with_punishment(sigma, n_train, n_fit, m_range, layout, ln_lam):
    # 用不同阶数分别拟合正弦函数（带惩罚项）
    # sigma为数据噪声的标准差
    # n_train为用来训练的数据集大小
    # n_fit为用来拟合的数据集大小
    # m_range为不同的阶数
    # layout为画图时排版格式（行数和列数）
    # ln_lam为超参数lambda取对数后的值

    fig, axes = plt.subplots(*layout)

    x_train, t_train = mT.generate_data_with_noise(sigma, n_train)
    x_fit = np.linspace(0, 1, n_fit)
    # 对每个order分别进行拟合并画图
    for i in range(len(m_range)):
        m = m_range[i]
        # 画图操作
        # 确定画图位置
        ax_target = axes[i // layout[1]][i % layout[1]]
        # 画训练集中各点
        # 默认情况下绘制线性回归拟合直线，用fit_reg = False将其删除
        sns.regplot(x=x_train, y=t_train, fit_reg=False, color="b", label="training set", ax=ax_target)
        # 画待拟合函数
        sns.lineplot(x=x_fit, y=np.sin(2 * np.pi * x_fit), color="g", label="$\\sin(2\\pi x)$", ax=ax_target)
        # 画多项式拟合曲线
        analytical_solution = AnalyticalSolution(mT.get_matrix_x(x_train, m), t_train.T)
        sns.lineplot(x=x_fit, y=mT.get_predictive_y(x_fit, analytical_solution.with_punishment(np.exp(ln_lam)), m),
                     color="r", label="fit result", ax=ax_target)
        # 设置图名
        title = "M = " + str(m) + ", N_train = " + str(n_train)
        props = {'title': title}
        ax_target.set(**props)

    plt.show()


def different_n_train_with_punishment(sigma, n_train_range, n_fit, m, layout, ln_lam):
    # 用不同训练集大小分别训练，以拟合正弦函数（带惩罚项）
    # sigma为数据噪声的标准差
    # n_train_range为不同的用来训练的数据集大小
    # n_fit为用来拟合的数据集大小
    # m为拟合时的多项式的阶数
    # layout为画图时排版格式（行数和列数）
    # ln_lam为超参数lambda取对数后的值

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
        analytical_solution = AnalyticalSolution(mT.get_matrix_x(x_train, m), t_train.T)
        sns.lineplot(x=x_fit, y=mT.get_predictive_y(x_fit, analytical_solution.with_punishment(np.exp(ln_lam)), m),
                     color="r", label="fit result", ax=ax_target)
        # 设置图名
        title = "M = " + str(m) + ", N_train = " + str(n_train)
        props = {'title': title}
        ax_target.set(**props)

    plt.show()
