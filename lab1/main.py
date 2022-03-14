import numpy as np

import myTool as mT
import analyticalSolution as aS
import gradientDescent as gD
import conjugateGradient as cG

if __name__ == '__main__':

    N_train = 10  # 训练集的大小
    N_exact = 1000  # 从正弦函数上所取样本的数量
    sigmaRange = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])  # 不同噪声标准差
    layout = (3, 3)  # 画图时的排版格式
    # 展示不同标准差下，生成的数据情况
    mT.different_sigma(sigmaRange, N_train, N_exact, layout)

    # 不带惩罚项的解析解(不同阶数、不同数据量)
    sigma = 0.3  # 噪声的标准差
    N_fit = 1000  # 用于拟合的数据集的大小
    mRange = range(1, 10)  # 拟合所用多项式的不同阶数
    # 展示相同N_train,不同m时的拟合曲线
    aS.different_m(sigma, N_train, N_fit, mRange, layout)
    N_trainRange = np.array([10, 50, 100, 150, 200, 250, 300, 500, 1000])  # 训练集的不同大小
    m = 9  # 拟合所用多项式的阶数
    # 展示相同m,不同N_train时的拟合曲线
    aS.different_n_train(sigma, N_trainRange, N_fit, m, layout)

    # 带惩罚项的解析解(不同超参数、不同阶数、不同数据量)
    N_test = 1000  # 测试集的大小
    ln_lam_Range = range(-50, 1, 1)  # 超参数的不同取值
    # 展示不同lambda时，训练集E_rms与测试集E_rms的变化情况(m = 9, N_train = 10, n_test = 1000)
    aS.different_lam_e_rms(sigma, N_train, N_test, ln_lam_Range)
    # 进行500次实验，寻找拟合最佳的lambda
    aS.find_lam(sigma, N_train, N_test, ln_lam_Range, 500)
    ln_lam = -8
    # 展示m = 9, N_train = 10, lambda = e^(-8)时，有惩罚项和无惩罚项的拟合曲线
    aS.with_and_no_punishment(sigma, N_train, N_test, ln_lam)
    # 展示相同N_train,相同lambda,不同m时的拟合曲线
    aS.different_m_with_punishment(sigma, N_train, N_fit, mRange, layout, ln_lam)
    # 展示相同m,相同lambda,不同N_train时的拟合曲线
    aS.different_n_train_with_punishment(sigma, N_trainRange, N_fit, m, layout, ln_lam)

    # 梯度下降(不同超参数、不同阶数、不同数据量)
    alphaRange = np.array([0.126, 0.127, 0.128])  # 超参数的不同取值
    m = 3
    # 展示不同alpha时，E_rms随迭代次数的变化情况(m = 3, N_train = 10)
    gD.different_alpha_e_rms(sigma, N_train, m, -8, alphaRange)
    # 展示m = 3, N_train = 10, lambda = e^(-8), alpha = 0.128时，梯度下降和有惩罚项的拟合曲线
    gD.gradient_descent_and_with_punishment(sigma, N_train, N_test, m, ln_lam, 0.128)
    mRange = np.array([3, 5, 7, 9])
    layout = (2, 2)
    # 展示相同N_train,相同alpha,不同m时的拟合曲线
    gD.different_m(sigma, N_train, N_fit, mRange, ln_lam, 0.01, layout)
    N_trainRange = np.array([10, 30, 50, 70])
    # 展示相同m,相同alpha,不同N_train时的拟合曲线
    gD.different_n_train(sigma, N_trainRange, N_fit, m, ln_lam, 0.01, layout)

    # 共轭梯度(不同阶数、不同数据量)
    # 展示m = 3, N_train = 10, lambda = e^(-8), alpha = 0.128时，梯度下降和共轭梯度的拟合曲线
    cG.gradient_descent_and_conjugate_gradient(sigma, N_train, N_test, m, ln_lam, 0.128)
    # 展示相同N_train,不同m时的拟合曲线
    cG.different_m(sigma, N_train, N_fit, mRange, ln_lam, layout)
    # 展示相同m,不同N_train时的拟合曲线
    cG.different_n_train(sigma, N_trainRange, N_fit, m, ln_lam, layout)
