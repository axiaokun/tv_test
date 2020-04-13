import numpy as np


class LWLR_model:
    def __init__(self):
        self.X = None
        self.y = None

    def _LWLR(self, text_x, k):
        """
        对每一个点进行预测
        :param text_x:训练数据，也就参考点 (array,2D)
        :param k:衰减函数中的波长参数 (float)
        :return:单个样本的预测值 (float)
        """
        n_sample = self.X.shape[0]
        weight = np.mat(np.zeros((n_sample, n_sample)))  # 创建权值矩阵
        for i in range(n_sample):
            diff = text_x - self.X[i, :]
            weight[i, i] = np.exp(diff * diff.T / (-2.0 * k ** 2))  # 根据预测点与参考点的距离，用衰减函数计算每个点的权值
        theta = np.linalg.inv(self.X.T.dot(weight.dot(self.X))).dot(self.X.T.dot(weight.dot(self.y)))  # 计算参数
        return text_x.dot(theta)  # 返回该预测点的预测值

    def fit(self, text_X, X, y, k=1.0):
        """
        拟合，预测数据
        :param text_X:训练数据 (array,2D)
        :param X:测试数据 (array,2D)
        :param y:训练数据对应目标值 (array,1D)
        :param k:衰减函数中的波长参数 (float)
        :return:样本的预测值 (array)
        """
        text_X = np.c_[np.ones((text_X.shape[0], 1)), text_X]  # 加入偏置项
        self.X = np.mat(np.c_[np.ones((X.shape[0], 1)), X])
        self.y = np.mat(y).T
        n_sample = self.X.shape[0]
        y_predict = np.zeros((n_sample,))
        for i in range(n_sample):
            y_predict[i] = self._LWLR(text_X[i], k)
        return y_predict