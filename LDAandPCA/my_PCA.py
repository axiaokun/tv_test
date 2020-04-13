# encode: utf-8
import numpy as np


class my_PCA:
    def __init__(self, k):
        self.k = k

    def pca(self, X):
        """

        :param X: Array over two dimensions
        :return:Array after dimension reduction
        """
        feature_mean = np.mean(X, axis=0)  # 计算各列的平均值，也就是各个特征的平均值
        out_mean = X - feature_mean  # 去中心化
        cov_matrix = out_mean.T.dot(out_mean)/X.shape[0]  # 协方差矩阵
        e_values, e_vectors = np.linalg.eig(cov_matrix)  # 计算特征值和特征向量
        min_sort = np.argsort(e_values)  # 获得由小到大的样本的标签
        k_max = min_sort[:-self.k-1:-1]  # 获取前k大的标签
        feature = e_vectors.T[k_max].T  # 获得对应的特征向量
        return X.dot(feature)


