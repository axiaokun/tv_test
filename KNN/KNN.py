import numpy as np
from collections import Counter


class KNNClassifier(object):
    """模拟KNN分类器"""
    def __init__(self, k):
        """
        :param k: 近邻数
        """
        assert k >= 1
        self.k = k
        self.x_train = None
        self.y_train = None

    def fit(self, x_train, y_train):
        """
        拟合数据
        :param x_train: 训练数据集，类型为二维数组
        :param y_train: 样本标签，一维数组
        :return: None
        """
        self.x_train = x_train
        self.y_train = y_train

    def _predict(self, x):
        """
        给定单个待预测数据x，返回x的预测结果值
        :param x: 单个数据样本，类型为一维数组
        :return: 最近的k个近邻中的类别以及数量
        """
        distances = []
        for i in range(len(self.x_train)):
            distance = np.linalg.norm(x-self.x_train[i], 2)
            distances.append((distance, self.y_train[i]))
        distances.sort()   # 这里的sort会根据元素的头一数据排序
        neightbors = distances[:self.k]
        target = [i[-1] for i in neightbors]
        return Counter(target).most_common()[0][0]  # 取最多的类别的标签

    def predict(self, X_predict):
        """
        :param X_predict: 待预测数据集，类型为二维数组
        :return: 表示X_predict的结果向量
        """
        return [self._predict(i) for i in X_predict]



