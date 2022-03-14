import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse

import kmeans


def plot_clusters(data, mu, sigma, mu_true=None, sigma_true=None):
    """
    画出各类的实际边界和预测边界
    :param data: 待分类数据集
    :param mu: 预测的各类的均值
    :param sigma: 预测的各类的标准差
    :param mu_true: 实际的各类的均值
    :param sigma_true: 实际的各类的标准差
    :return: 数据集、实际边界和预测边界的图
    """
    cluster_num = len(mu)
    # 画出数据集
    plt.scatter(data[:, 0], data[:, 1])
    ax = plt.gca()
    # 画出预测边界
    for i in range(cluster_num):
        plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': 'g', 'ls': ':'}
        ellipse = Ellipse(mu[i], 3 * math.sqrt(sigma[i][0]), 3 * math.sqrt(sigma[i][1]), **plot_args)
        ax.add_patch(ellipse)
    # 画出实际边界
    if (mu_true is not None) and (sigma_true is not None):
        for i in range(cluster_num):
            plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': 'b'}
            ellipse = Ellipse(mu_true[i], 3 * math.sqrt(sigma_true[i][0]), 3 * math.sqrt(sigma_true[i][1]), **plot_args)
            ax.add_patch(ellipse)
    plt.show()


def gmm(cluster_num, data, times, show=True, get_cluster_centers=False):
    """
    用GMM对数据集分类
    :param cluster_num: 类的数量
    :param data: 待分类数据集
    :param times: 迭代次数
    :param show: 是否展示对数似然函数随迭代次数的变化情况图
    :param get_cluster_centers: 是否返回最终各类中心
    :return: 对各数据的分类标签（及最终各类中心）
    """
    dim, num = data.shape[1], data.shape[0]

    # 用kmeans结果作为初始化均值与标准差
    label = kmeans.k_means(data, cluster_num, 10)
    mus = np.zeros((cluster_num, dim))
    sigmas = np.zeros((cluster_num, dim))
    cluster = []
    for i in range(cluster_num):
        for j in range(num):
            if label[j] == i:
                cluster.append(data[j])
        mus[i, :] = np.average(cluster, axis=0)
        cluster = np.array(cluster)
        sigmas[i, :] = np.average((cluster - mus[i])**2, axis=0)
        cluster = []

    # 选定最初参数
    mu = mus + 2 * np.random.randn(cluster_num, dim)
    sigma = sigmas + abs(2 * np.random.randn(cluster_num, dim))
    gama_matrix = np.ones((num, cluster_num)) / cluster_num
    pi = gama_matrix.sum(axis=0) / gama_matrix.sum()

    log_lh = []
    for i in range(times):
        # 展示每次迭代的效果图
        # plot_clusters(data, mu, sigma, mus, sigmas)
        # 计算对数似然函数值，并更新参数
        log_lh.append(cal_log_lh(data, pi, mu, sigma))
        gama_matrix = cal_matrix(data, mu, sigma, pi)
        pi = cal_pi(gama_matrix)
        mu = cal_mu(data, gama_matrix)
        sigma = cal_sigma(data, mu, gama_matrix)
        print('log-likehood:%.5f' % log_lh[-1])

    if show:
        plt.plot(log_lh)
        plt.title("log-likehood changed graph")
        plt.show()

    # 对数据集进行分类
    label = np.zeros(num)
    for xi in range(num):
        probability = np.zeros(cluster_num)
        for i in range(cluster_num):
            probability[i] = multivariate_normal.pdf(data[xi, :], mu[i], np.diag(sigma[i]))
        label[xi] = np.argmax(probability)

    if not get_cluster_centers:
        label = label.reshape(-1, 1)
        return label
    else:
        return label, mu


def cal_log_lh(data, pi, mu, sigma):
    """
    计算对数似然函数值
    :param data: 待分类数据集
    :param pi: 公式中pi值
    :param mu: 公式中mu值
    :param sigma: 公式中sigma值
    :return: 对数似然函数值
    """
    num, cluster_num = len(data), len(pi)
    pdfs = np.zeros((num, cluster_num))
    for i in range(cluster_num):
        pdfs[:, i] = pi[i] * multivariate_normal.pdf(data, mu[i], np.diag(sigma[i]))
    return np.sum(np.log(pdfs.sum(axis=1)))


def cal_matrix(data, mu, sigma, pi):
    """
    计算公式中的gama矩阵
    :param data: 待分类数据集
    :param mu: 公式中mu值
    :param sigma: 公式中sigma值
    :param pi: 公式中pi值
    :return: 公式中的gama矩阵
    """
    num, cluster_num = len(data), len(pi)
    pdfs = np.zeros((num, cluster_num))
    for i in range(cluster_num):
        pdfs[:, i] = pi[i] * multivariate_normal.pdf(data, mu[i], np.diag(sigma[i]))
    gama_matrix = pdfs / pdfs.sum(axis=1, keepdims=True)
    return gama_matrix


def cal_pi(gama_matrix):
    """
    计算公式中的pi
    :param gama_matrix: 公式中gama矩阵
    :return: 公式中的pi
    """
    n = gama_matrix.shape[0]
    pi = gama_matrix.sum(axis=0) / n
    return pi


def cal_mu(data, gama_matrix):
    """
    计算公式中的mu
    :param data: 待分类数据集
    :param gama_matrix: 公式中gama矩阵
    :return: 公式中的mu
    """
    dim, cluster_num = data.shape[1], gama_matrix.shape[1]
    mu = np.zeros((cluster_num, dim))
    for i in range(cluster_num):
        mu[i, :] = np.average(data, axis=0, weights=gama_matrix[:, i])
    return mu


def cal_sigma(data, mu, gama_matrix):
    """
    计算公式中的sigma
    :param data: 待分类数据集
    :param mu: 公式中mu
    :param gama_matrix: 公式中gama矩阵
    :return: 公式中的sigma
    """
    dim, cluster_num = data.shape[1], gama_matrix.shape[1]
    sigma = np.zeros((cluster_num, dim))
    for i in range(cluster_num):
        sigma[i, :] = np.average((data - mu[i]) ** 2, axis=0, weights=gama_matrix[:, i])
    return sigma
