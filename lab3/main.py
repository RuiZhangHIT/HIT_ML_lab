import numpy as np

import mytool as mt
import kmeans
import gmm

if __name__ == '__main__':

    # 用K_means对数据集分类
    np.random.seed(0)
    times = 10
    mus = [[1, 1], [1, 3], [3, 1], [3, 3]]
    sigmas = [[0.8, 0.8], [0.8, 0.8], [0.9, 0.9], [0.9, 0.9]]
    nums = [200, 200, 200, 200]
    data = mt.generate_data(mus, sigmas, nums, 2)
    labels_pre = kmeans.k_means(data, 4, times)
    mt.color_data(np.concatenate([data, labels_pre], axis=1))

    # 用GMM对数据集分类（生成的数据）
    np.random.seed(0)
    times = 10
    mus = [[1, 1], [2, 3], [3, 2]]
    sigmas = [[1, 1], [0.8, 0.8], [0.7, 0.7]]
    nums = [200, 200, 200]
    data = mt.generate_data(mus, sigmas, nums, 2)
    labels_pre = gmm.gmm(3, data, times)
    mt.color_data(np.concatenate([data, labels_pre], axis=1))

    # 用GMM对数据集分类（UCI上的数据）
    np.random.seed(0)
    times = 10
    data, labels_real = mt.load_data("iris.data")
    labels_pre, mu = gmm.gmm(3, data, times, show=True, get_cluster_centers=True)
    print("real labels:", labels_real)
    print("prediction labels:", labels_pre)
    print("cluster centers:", mu)
    print("accuracy:", mt.cal_accuracy(labels_pre, labels_real))
