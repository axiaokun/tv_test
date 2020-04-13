import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans


class Kmean_simple:
    """一个简单的K_mean模型"""
    def __init__(self, k=2, max_iterations=500):
        self.k = k
        self.max_iterations = max_iterations

    def _init_centers(self, X):
        """
        创建第一批中心点
        :param X: 训练数据，，类型为二维数组
        :return: 簇中心组成的列表
        """
        n_sample, n_featrue = X.shape[0], X.shape[1]
        centers = np.zeros((self.k, n_featrue))
        for i in range(self.k):
            center = X[np.random.randint(n_sample)]  # 随机在数据集中挑选k个簇中心
            centers[i] = center
        return centers

    def _create_complexs(self, X, centres):
        """
        创建各个类的集合体，也就是划分各个簇
        :param X: 训练数据，类型为二维数组
        :param centres: 簇中心
        :return: 各个簇的集合体，每个簇中包含属于这个簇的数据在X中的index
        """
        complexs = [[] for _ in range(self.k)]
        for i_index, i_sample in enumerate(X):
            init_distance = float('inf')
            for j_index, j_center in enumerate(centres):  # 划分簇集合
                distance = np.linalg.norm(i_sample - j_center)
                if distance < init_distance:
                    init_distance = distance
                    sample_type = j_index  # 取距离最短的簇中心所在簇作为该点的簇
            complexs[sample_type].append(i_index)
        return complexs

    def _recenters(self, X, complexs):
        """
        重新修改中心点
        :param X: 训练数据，类型为二维数组
        :param complexs: 各个簇的集合体
        :return: 重新划分后的簇中心
        """
        n_featrue = X.shape[1]
        centers = np.zeros((self.k, n_featrue))
        for type_index, type_value in enumerate(complexs):
            new_center = np.mean(X[type_value], axis=0)  # 重新计算簇中心，取簇中数据的平均值
            centers[type_index] = new_center
        return centers

    def _get_results(self, X, complexs):
        """
        得出最后的结果
        :param X: 数据集，类型为二维数组
        :param complexs: 簇集合体
        :return: 聚类结果
        """
        result_predict = np.zeros((X.shape[0], ))
        for type_index, type_value in enumerate(complexs):
            for sample_index in type_value:
                result_predict[sample_index] = type_index
        return result_predict

    def predict(self, X):
        """
        进行聚类主函数
        :param X: 数据集，类型为二维数据
        :return: 聚类结果
        """
        centres = self._init_centers(X)  # 初始化簇中心
        time = 0
        while time < self.max_iterations:
            time += 1
            complexs = self._create_complexs(X, centres)  # 划分簇集合
            pre_centers = centres
            centres = self._recenters(X, complexs)  # 重新确立簇中心
            if np.linalg.norm(centres-pre_centers) == 0:  # 一旦簇中心的位置不变，那么停止迭代
                return self._get_results(X, complexs)
