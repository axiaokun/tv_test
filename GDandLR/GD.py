import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


class SGD_simple():

    def __init__(self, eta=0.01, n_iterations=50000, epsilon=1e-5):
        """
        :param eta:步长
        :param n_iterations:最大迭代次数
        :param epsilon: 允许最大误差
        """
        self.eta = eta
        self.n_iterations = n_iterations
        self.epsilon = epsilon
        self.fit_theta = None
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        """
        拟合数据
        :param X: 训练数据，类型为二维数组
        :param y: 样本标签，类型为一维数组
        :return: None
        """
        iteration = 0
        loss = 1
        y = y.reshape((len(y), 1))
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # 这里要加入一列，作为截距
        theta = np.random.rand(X_b.shape[1], 1)  # 初始化系数
        sample = X_b.shape[0]
        while iteration < self.n_iterations and loss > self.epsilon:
            last_theta = theta
            iteration += 1
            random_index = np.random.randint(sample)
            X_i = X_b[random_index:random_index+1]  # 随机选择样本
            y_i = y[random_index:random_index+1]
            gradients = 2 * X_i.T.dot(X_i.dot(theta) - y_i)  # 计算梯度
            theta = theta - self.eta * gradients  # 更新梯度
            loss = np.linalg.norm(theta - last_theta)
        self.fit_theta = theta
        self.coef_ = theta[1:].reshape(X.shape[1])   # 系数权重
        self.intercept_ = theta[:1].reshape(1)  # 截距

    def predict(self, X_i):
        """
        计算预测结果
        :param X_i:待预测的数据集
        :return: 预测结果
        """
        X_b = np.c_[np.ones((X_i.shape[0], 1)), X_i]
        return X_b.dot(self.fit_theta)


class BGD_simple:
    def __init__(self, eta=0.01, n_iterations=10000, epsilon=1e-5):
        """
        :param eta:步长
        :param n_iterations:最大迭代次数
        :param epsilon: 允许最大误差
        """
        self.eta = eta
        self.n_iterations = n_iterations
        self.epsilon = epsilon
        self.fit_theta = None
        self.coef_ = None  # 参数的权重系数
        self.intercept_ = None  # 截距

    def fit(self, X, y):
        """
        拟合数据
        :param X: 训练数据，类型为二维数组
        :param y: 样本标签，类型为一维数组
        :return: None
        """
        iteration = 0
        loss = 1
        y = y.reshape((len(y), 1))
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # 加入一列截距
        theta = np.random.rand(X_b.shape[1], 1)  # 初始化权重系数
        sample = X_b.shape[0]  # 样本个数
        while iteration < self.n_iterations and loss > self.epsilon:
            last_theta = theta
            iteration += 1
            gradients = 2/sample * X_b.T.dot(X_b.dot(theta) - y)  # 计算梯度
            theta = theta - self.eta * gradients  # 更新权重
            loss = np.linalg.norm(theta - last_theta)  # 计算两次权重之间的差距
        self.fit_theta = theta
        self.coef_ = theta[1:].reshape(X.shape[1])
        self.intercept_ = theta[:1].reshape(1)

    def predict(self, X_i):
        """
        计算预测结果
        :param X_i:待预测的数据集
        :return: 预测结果
        """
        X_b = np.c_[np.ones((X_i.shape[0], 1)), X_i]  # 增加一列截距
        return X_b.dot(self.fit_theta)  # 与权重系数相乘得出预测结果


class MBGD_simple:
    def __init__(self, eta=0.01, n_iterations=10000, epsilon=1e-5, batch_size=2):
        """
        :param eta:步长
        :param n_iterations:最大迭代次数
        :param epsilon: 允许最大误差
        :param batch_size: 一个批次的数量
        """
        self.eta = eta
        self.n_iterations = n_iterations
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.fit_theta = None
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        """
        拟合数据
        :param X: 训练数据，类型为二维数组
        :param y: 样本标签，类型为一维数组
        :return: None
        """
        iteration = 0
        loss = 1
        y = y.reshape((len(y), 1))
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        theta = np.random.rand(X_b.shape[1], 1)
        sample = X_b.shape[0]
        while iteration < self.n_iterations and loss > self.epsilon:
            last_theta = theta
            iteration += 1
            i = np.random.randint(sample)  # 随机选择
            i_batch_size = (i + self.batch_size) % (sample)  # 取随机选择的数字后一个批次的数量
            if i + self.batch_size > sample-1:  # 如果大于样本数量则分两段处理
                for j in range(i, sample):
                    X_i = X_b[j: j + 1]
                    y_i = y[j: j + 1]
                    gradients = 2 * X_i.T.dot(X_i.dot(theta) - y_i)  # 计算梯度
                    theta = theta - self.eta * gradients  # 更新权重
                for j in range(i_batch_size):
                    X_i = X_b[j: j + 1]
                    y_i = y[j: j + 1]
                    gradients = 2 * X_i.T.dot(X_i.dot(theta) - y_i)
                    theta = theta - self.eta * gradients
            else:
                for j in range(i, i_batch_size):
                    X_i = X_b[j: j + 1]
                    y_i = y[j: j + 1]
                    gradients = 2 * X_i.T.dot(X_i.dot(theta) - y_i)
                    theta = theta - self.eta * gradients
            loss = np.linalg.norm(theta - last_theta)
        self.fit_theta = theta
        self.coef_ = theta[1:].reshape(X.shape[1])
        self.intercept_ = theta[:1].reshape(1)

    def predict(self, X_i):
        """
        计算预测结果
        :param X_i:待预测的数据集
        :return: 预测结果
        """
        X_b = np.c_[np.ones((X_i.shape[0], 1)), X_i]
        return X_b.dot(self.fit_theta)


class LinearRegression_simple():

    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.fit_theta = None

    def fit(self, X, y):
        """
        拟合数据
        :param X:训练数据，类型为二维数组
        :param y: 样本标签，类型为一维数组
        :return: None
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        y = y.reshape(len(y), 1)
        self.fit_theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)  # 这里使用正规方程的公式进行计算权值系数
        self.coef_ = self.fit_theta[1:].reshape(X.shape[1])
        self.intercept_ = self.fit_theta[0].reshape(1)

    def predict(self, X_i):
        X_b = np.c_[np.ones((X_i.shape[0], 1)), X_i]
        return X_b.dot(self.fit_theta)

