import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from PIL import Image


def generate_data(mus, sigmas, nums, dim):
    """
    生成高斯分布的数据
    :param mus: 高斯分布的均值
    :param sigmas: 高斯分布的标准差
    :param nums: 高斯分布的个数
    :param dim: 数据的维度
    :return: 生成的数据集
    """
    datasets = []
    for mu, sigma, num in zip(mus, sigmas, nums):
        dataset = np.random.randn(num, dim)
        for i, v in enumerate(sigma):
            dataset[:, i] *= sigma[i]
        for i, m in enumerate(mu):
            dataset[:, i] += mu[i]
        datasets.append(dataset)
    datasets = np.concatenate(datasets)
    return np.array(datasets, dtype=float)


def rotate(data, theta=0, axis='x'):
    """
    将数据进行旋转
    :param data: 待旋转数据
    :param theta: 旋转角度
    :param axis: 旋转轴
    :return: 旋转后的数据集
    """
    if axis == 'x':
        r = [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]]
    elif axis == 'y':
        r = [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
    else:
        r = [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    return np.dot(r, data)


def show_pca_result(data, reconstructed_data):
    """
    以3D图展示PCA降维后的数据与原始数据
    :param data: 原始数据
    :param reconstructed_data: 降维后的数据
    :return: 数据的三视图
    """
    # 降维前后3D图
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c="r", label='Origin Data')
    ax.scatter(reconstructed_data[:, 0], reconstructed_data[:, 1], reconstructed_data[:, 2], c='g',
               label='Reconstructed Data')
    ax.plot_trisurf(reconstructed_data[:, 0], reconstructed_data[:, 1], reconstructed_data[:, 2], color='k', alpha=0.3)
    ax.legend(loc='best')
    plt.show()
    plt.style.use('default')
    # 三视图
    for elev, azim, title in zip([0, 0, 90], [0, 90, 0], ["yOz", "xOz", "xOy"]):
        fig = plt.figure()
        fig.suptitle(title)
        ax = Axes3D(fig)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.view_init(elev=elev, azim=azim)
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c="r", label='Origin Data')
        ax.scatter(reconstructed_data[:, 0], reconstructed_data[:, 1], reconstructed_data[:, 2], c='g',
                   label='Reconstructed Data')
        ax.plot_trisurf(reconstructed_data[:, 0], reconstructed_data[:, 1], reconstructed_data[:, 2], color='k',
                        alpha=0.3)
        ax.legend(loc='best')
        plt.show()
        plt.style.use('default')


def load_faces():
    """
    读入初始人脸数据，并展示初始图像
    :return: 人脸数据组成的矩阵
    """
    data = []
    for i in range(8):
        img = Image.open('./faces/' + str(i + 1) + '.jpg')
        img = np.array(img)
        plt.subplot(2, 4, i + 1)
        plt.title("face:" + str(i + 1))
        plt.xlabel("Original")
        plt.imshow(img)
        data.append(img.reshape(50 * 50))
    plt.show()
    return np.array(data)


def show_compressed_faces(data, reconstructed_data):
    """
    展示PCA处理后的人脸图像，并计算信噪比
    :param data: 处理前数据
    :param reconstructed_data: 降维后数据
    :return: 处理后人脸图像
    """
    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.title("face:" + str(i + 1))
        plt.xlabel("PSNR:%.3fdB" % cal_psnr(data[i], reconstructed_data[i]))
        plt.imshow(reconstructed_data[i].astype(np.float).reshape((50, 50)))
    plt.show()


def cal_psnr(origin, compress):
    """
    计算图像信噪比
    :param origin: 原始图像
    :param compress: 压缩后图像
    :return: 图像信噪比
    """
    mse = np.mean((compress - origin) ** 2)
    psnr = 20 * math.log10(255 / math.sqrt(mse))
    return psnr
