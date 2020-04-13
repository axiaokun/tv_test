# encode: utf-8
import numpy as np
from sklearn.datasets import load_iris


def my_lda(X, y):
    """

    :param X: array with the 2D shape
    :param y: array with the 1D shape
    :return:
    """
    data_0, data_1 = X[(y == 0)], X[(y != 0)]
    mean_0, mean_1 = np.mean(data_0, axis=0), np.mean(data_1, axis=0)
    out_mean_0, out_mean_1 = data_0-mean_0, data_1-mean_1  # 去均值
    sw = out_mean_0.T.dot(out_mean_0) + out_mean_1.T.dot(out_mean_1)  # 计算类内散度矩阵
    w = (mean_0 - mean_1).dot(np.linalg.inv(sw))  # 拉格朗日乘子法，计算出w
    return w


