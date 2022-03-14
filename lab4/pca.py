import numpy as np


def pca(data, k):
    """
    用PCA对数据降维
    :param data: 待降维数据集
    :param k: 需要降到的维度
    :return: 降维后的数据以及恢复后的数据
    """
    m, n = np.shape(data)
    average = np.mean(data, axis=0)
    average_matrix = np.tile(average, (m, 1))
    data_adjust = data - average_matrix
    cov_matrix = np.cov(data_adjust.T)
    values, vectors = np.linalg.eig(cov_matrix)
    index = np.argsort(-values)
    if k > n:
        print("k should be smaller than the featrue")
        return
    else:
        select_vectors = vectors[:, index[:k]]
        select_vectors = np.real(select_vectors)  # 对于带实数的特征向量矩阵，保留实部即可
        compressed_data = np.dot(data_adjust, select_vectors)
        reconstructed_data = np.dot(compressed_data, select_vectors.T) + average
    return compressed_data, reconstructed_data
