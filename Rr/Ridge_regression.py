import numpy as np


class LinearRegression_simple:
    def __init__(self, alphe=0.01):
        """
        :param alphe:正则系数 (float)
        """
        self.coef_ = None
        self.intercept_ = None
        self.fit_theta = None
        self.alphe = alphe

    def fit(self, X, y):
        """
        :param X: 训练数据 (array,2D)
        :param y: 训练样本目标值 (array,1D)
        :return: None
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        n_sample, n_featrue = X_b.shape
        y = y.reshape(len(y), 1)
        panish = np.zeros((n_featrue, n_featrue))
        for i in range(1, n_featrue):
            panish[i][i] = 1  # 构造一个向量空间，正对角线上第一个位置为0，其他位置为1；非正对角线位置全为0

        # 利用公式计算权重，类似正规方程，具体推导见md文件
        self.fit_theta = np.linalg.inv(X_b.T.dot(X_b) + self.alphe * panish).dot(X_b.T).dot(y)
        self.coef_ = self.fit_theta[1:].reshape(X.shape[1])
        self.intercept_ = self.fit_theta[0].reshape(1)

    def predict(self, X_i):
        """
        :param X_i: 测试数据 (array, 2D)
        :return: 预测结果
        """
        X_b = np.c_[np.ones((X_i.shape[0], 1)), X_i]
        return X_b.dot(self.fit_theta)