import numpy as np

import mytool as mt
import logisticregression as lr


if __name__ == '__main__':
    np.random.seed(0)

    class_0_number = 200  # 类0的数据集大小（含训练集与测试集）
    class_1_number = 200  # 类1的数据集大小（含训练集与测试集）
    eta = 0.48  # 梯度下降求解时的步长
    train_rate = 0.7  # 训练集大小占数据集大小的比例
    times = 150000  # 梯度下降求解时的迭代次数上限

    # 满足朴素贝叶斯的数据集
    # x_train, y_train, x_test, y_test = mt.generate_data(1, 3, 0.6, class_0_number, class_1_number, train_rate)
    # 不满足朴素贝叶斯的数据集
    # x_train, y_train, x_test, y_test = mt.generate_data(1, 3, 0.6, class_0_number, class_1_number, train_rate, cov=0.4)
    # 使用UCI上的数据
    x_train, y_train, x_test, y_test = mt.load_data("iris.data", train_rate)

    # 初始化逻辑回归分类器
    classifier = lr.LogisticRegression(len(x_train[0]))

    # 损失函数无惩罚项
    # loss_list, t = classifier.solve(x_train, y_train, eta, times)
    # 损失函数带惩罚项
    loss_list, t = classifier.solve(x_train, y_train, eta, times, lam=1e-8)

    # 展示梯度下降法求得的w和b，最终结果下训练集的代价函数值，以及该分类器在测试集上的准确率
    print("w of the classifier:", classifier.w)
    print("b of the classifier:", classifier.b)
    print("accuracy of the test set:", classifier.accuracy(x_test, y_test))

    # 画出决策边界与测试集分布情况
    x_all = np.concatenate([x_train[:, 0], x_test[:, 0]])
    classifier.draw_border(min(x_all), max(x_all))

    # 画出loss随迭代次数的变化情况
    mt.draw_loss_line(t, loss_list)
