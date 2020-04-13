import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


class Lg:

    def __init__(self, learning_rate=0.2, epochs=1000):
        """
        :param learning_rate:学习率
        :param epochs: 最大迭代次数
        """
        self.lr = learning_rate
        self.epochs = epochs
        self.pi_list = []
        self.weights = None

    def _sigmoid(self, x):
        return np.exp(x) / (1 + np.exp(x))

    def fit(self, X, y):
        """
        拟合数据
        :param X:训练数据集，二维数组
        :param y: 样本标签，一维数组
        :return: None
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        weights = np.random.randn(X_b.shape[1], 1)
        for _ in range(self.epochs):  # 注意这里使用的是梯度上升法计算使得似然函数最大时权值系数的估计值
            i = np.random.randint(X.shape[0])
            X_i = np.mat(X_b[i])
            pi = self._sigmoid(X_i.dot(weights))
            gradient = (y[i] - pi) * X_i
            weights += self.lr * gradient.T
        self.weights = weights

    def predict(self, X):
        """
        进行预测
        :param X: 训练数据，类型为二维数组
        :return: 预测结果，一维数组
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        result = []
        for X_i in X_b:
            pi = self._sigmoid(X_i.dot(self.weights))
            self.pi_list.append(pi[0])
            result.append(round(float(pi)))  # 将预测sigmoid函数计算得到的结果转化为整数后加入数组中
        return np.array(result)


if __name__ == '__main__':
    digits = load_digits()
    data_X, data_y = digits.data, digits.target
    std = StandardScaler()
    data_X = std.fit_transform(data_X)
    Lg_model = Lg()
    Lg_model.fit(data_X, data_y)
