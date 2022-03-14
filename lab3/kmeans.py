import numpy as np
import math


def k_means(data, cluster_num, times):
    """
    用K_means求解数据集分类问题
    :param data: 待分类的数据集
    :param cluster_num: 类的数量
    :param times: 迭代次数
    :return: 对各数据的分类标签
    """

    # 从样本中随机选择初始时的类中心
    center = np.zeros((cluster_num, data.shape[1]))
    for num in range(cluster_num):
        center[num, :] = data[np.random.randint(0, data.shape[0]), :]

    new_center = np.zeros((center.shape[0], center.shape[1]))
    label = np.zeros(data.shape[0], )

    for i in range(times):
        distance = np.zeros(center.shape[0], )
        # 计算每个点到各个类中心的距离，并按最小距离进行分类
        for j in range(data.shape[0]):
            for k in range(center.shape[0]):
                d = data[j, :]
                c = center[k, :]
                distance[k] = math.sqrt(sum((d - c) ** 2))
            label[j] = np.argmin(distance)
        # 重新计算每一类的中心
        for k in range(center.shape[0]):
            cluster = []
            for j in range(data.shape[0]):
                if label[j] == k:
                    cluster.append(data[j, :])
            if not cluster:
                continue
            new_center[k, :] = np.average(cluster, axis=0)
        center = new_center

    label = label.reshape(-1, 1)
    return label
