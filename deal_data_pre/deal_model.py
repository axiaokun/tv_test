# -*-coding: utf-8-*-

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def deal_Discrete_values(data, columns_index=None):
    """
    独热编码

    :param data:数据，类型为DataFrame
    :param columns_index: 需要实现一种热编码的列标签列表
    :return: 进行独热编码后的数据

    如果参数列索引未传入，程序将查找非数值，单独对其执行独热编码

    :raise
    ValueError:传入的数据对象不是DataFrame类型
    AssertionError:指定的列不在传入的DataFrame数据中
    """

    if columns_index is None:
        return pd.get_dummies(data)  # 不指定列，自行进行独热编码

    if type(data) != pd.DataFrame:
        raise ValueError('Data of type pd.DataFrame is required')

    columns_index_data = data.columns.values
    assert set(columns_index) < set(columns_index_data), \
        'Columns that need to implement one hot encoding are not in this DataFrame'

    for i in columns_index:
        data = data.join(pd.get_dummies(data[i]))  # 指定列进行独热编码，并加入DataFrame中

    return data.drop(columns=columns_index)  # 删除原来进行独热编码的列（注意这个删除不会改变原来的数据），并返回数据


def normalization(X):
    """
    对数据进行归一化
    :param X:二维数组
    :return:归一化后的数据
    """
    if type(X) != np.ndarray:
        raise TypeError('need the arg of array')

    min_columns = X.min(axis=0)  # 参数axis设置为0，求出每一列的最小值
    max_columns = X.max(axis=0)
    columns, rows = X.shape[1], X.shape[0]
    for i in range(columns):
        X[:, i] = (X[:, i] - min_columns[i]) / (max_columns[i] - min_columns[i])  # 归一化
    return X


def standardization(X):
    """
    对数据进行标准化

    :param X:二维数组
    :return:标准化后的数据
    """
    if type(X) != np.ndarray:
        raise TypeError('need the arg of array')

    mean_columns = np.mean(X, axis=0)
    std_columns = np.std(X, axis=0)
    columns, rows = X.shape[1], X.shape[0]
    for i in range(columns):
        X[:, i] = (X[:, i] - mean_columns[i]) / std_columns[i]  # 标准化
    return X


def fill_missing_values(*args, fill_values=None):
    """
    填补缺失值
    :param args: {
                    第一个参数：DataFrame的数据
                    第二个参数：需要进行填补的列标签
                    }
    :param fill_values:填充的数据类型或要填充的字符。您可以选择三种类型，平均数，中位数，众数。
                        如果没有输入参数，则默认为无
    :return:填补后的数据
    """
    X, columns_index = args
    if fill_values is None:
        for i in columns_index:
            X[i].fillna("None", inplace=True)  # 填补缺失值
        return X
    if fill_values == 'mean':
        for i in columns_index:
            X[i].fillna(X[i].mean(), inplace=True)
        return X
    if fill_values == 'median':
        for i in columns_index:
            X[i].fillna(X[i].median(), inplace=True)
        return X
    if fill_values == 'mode':
        for i in columns_index:
            X[i].fillna(X[i].mode(), inplace=True)
        return X
    else:
        for i in columns_index:
            X[i].fillna(fill_values, inplace=True)
        return X


class Exception_handling:
    def __init__(self):
        self.data = None
        self.jud_bool = None
        self.columns_index = None

    def fit(self, data, columns_index):
        """
        找到异常值并打印异常值所在的样本
        :param data:样本数据，类型为DataFrame
        :param columns_index:指定的列标签
        :return:异常值所在的样本，如果没有，则返回none
        """
        self.data = data
        self.columns_index = columns_index
        jud_stand = (data[columns_index].mean() + 3 * data[columns_index].std(),  # 计算平均值的±3个标准差
                     data[columns_index].mean() - 3 * data[columns_index].std())
        self.jud_bool = (data[columns_index] > jud_stand[0]) | (data[columns_index] < jud_stand[1])  # 查找这个范围之外的数据
        if data[self.jud_bool].empty:
            print("No outliers")
        else:
            return data[self.jud_bool]  # 如果有返回这些异常值所在的行也就是包含异常值的样本

    def replace_value(self):
        """
        处理上一次fit查到的异常值
        :return:处理后的数据
        """
        # 这里采用nan替换，留到后面和缺失值一起处理
        self.data.replace(self.data[self.jud_bool][self.columns_index].tolist(), np.nan, inplace=True)
        return self.data


def discretization(series, bins):
    """
    连续值离散化
    :param series:数据类型是一维数组或series对象
    :param bins:整数或分位数的数组，即将数据分组
    :return:D离散化结果
    """
    cut_result = pd.qcut(series, q=bins)  # 切割数据，离散化
    return cut_result


def euclidean_distance_similarity(*args):
    """
    用欧氏距离表示相似性
    :param args: 两个一维数组或列表，数据类型为数值型
    :return: 相似度（用欧氏距离表示）
    """
    data1, data2 = args
    if type(data1) == list or type(data2) == list:
        data1, data2 = np.array(data1), np.array(data2)

    data1.astype(np.float64)
    data2.astype(np.float64)
    return np.linalg.norm(data1 - data2)  # 计算欧氏距离


def manhattan_distance_similarity(*args):
    """
    用曼哈顿距离表示相似性
    :param args: 两个一维数组或列表，数据类型为数值型
    :return: 相似度（以曼哈顿距离表示）
    """
    data1, data2 = args
    if type(data1) == list or type(data2) == list:
        data1, data2 = np.array(data1), np.array(data2)

    data1.astype(np.float64)
    data2.astype(np.float64)
    return np.linalg.norm(data1 - data2, ord=1)  # 计算曼哈顿距离


def cosine_similarity(*args):
    """
    计算余弦相似性
    :param args: 两个一维数组或列表，数据类型为数值型
    :return: 余弦相似度
    """
    data1, data2 = args
    data1, data2 = np.array(data1, dtype=np.float64), np.array(data2, dtype=np.float64)
    return (data2.dot(data1.T)) / (np.linalg.norm(data1) * np.linalg.norm(data2))  # 计算余弦相似度


def pearson_correlation(*args):
    """
    计算皮尔逊相关
    :param args: 一维或二维数组。数据类型为数值型
    :return: 皮尔逊相关度
    """
    matrix_data = np.vstack(args).astype(np.float64)  # 拼接传进来的数组
    return np.corrcoef(matrix_data)  # 计算皮尔逊相关度


def cross_validation(model, X, y, cv=4):
    """
    :param model:模型
    :param X:训练数据，类型为DataFrame
    :param y:训练集数据对应标签，类型为array
    :param cv:交叉验证数(int)
    :return:交叉验证后的平均精度（int）
    """
    X['y_ran'] = y  # 把对应的标签值加入一起打乱
    X = X.sample(frac=1)
    y = np.array(X.pop('y_ran'))  # 取出打乱后的y值
    fold = int(X.shape[0] / cv)  # 每折有多少个样本
    score_sum = 0

    for i in range(cv):
        start, end = i*fold, (i+1)*fold
        test_data, test_y = X[start:end], y[start:end]  # 划分验证集
        train_data, train_y = pd.concat([X[:start], X[end:]]), np.concatenate([y[:start], y[end:]])  # 划分训练集
        model.fit(train_data, train_y)
        result = model.predict(test_data)
        score_sum += accuracy_sco(test_y, result)  # 计算准确率
    return score_sum / cv  # 计算几折预测后平均准确率


def retention_method(X, y, test_size=0.2):
    """
    留出法划分数据集
    :param X:训练数据集(DataFrame)
    :param y:训练数据集对应的标签(one-dimensional array)
    :param test_size:留出的百分比(float)
    :return:训练集，训练集标签，测试集，测试集标签
    """
    X['y_ran'] = y  # 前面的打乱基本和交叉验证模块里的一样
    X = X.sample(frac=1)
    y = np.array(X.pop('y_ran'))
    tangent_point = int(X.shape[0] * (1-test_size))  # 下面划分对应的数据集
    X_train, X_test, y_train, y_test = X[:tangent_point], X[tangent_point:], y[:tangent_point], y[tangent_point:]
    return X_train, X_test, y_train, y_test


def bootstrap_method(X):
    """
    自助法划分数据
    :param X:需要处理的数据 (DataFrame)
    :return:处理得出的数据(DataFrame)
    """
    index_list = (np.random.randint(X.shape[0]) for _ in range(X.shape[0]))  # 随机挑选样本标签
    pd_data = pd.DataFrame()
    for index in index_list:
        pd_data = pd_data.append(X[index:index+1])  # 注意这里要赋值更新，不然pd_data一直是空
    return pd_data


def check_shape(param_1, param_2):
    """
    检查是否为数组以及数组的形状是否为一维
    :param param_1:一维数组
    :param param_2:一维数组
    :return:检查后的数组

    :raise
    TypeError: 出入数据不是数组
    ValueError:  数组形状不一致或是不是一维数组
    """
    if type(param_2) != np.ndarray or type(param_1) != np.ndarray:  # 检查是否为数组
        raise TypeError(' Expect the type of array ')
    if np.shape(param_1) != np.shape(param_2):  # 检查数组shape是否一致
        raise ValueError(' Found input variables with inconsistent numbers of samples ')
    try:  # 检查数组是否是一维的，这里考虑到numpy数组一维的shape有两种表示，所以加入一个异常处理
        shape_c1, shape_c2 = param_1.shape[1], param_2.shape[1]
        if param_1.shape[0] != 1 or param_2.shape[0] != 1:
            raise ValueError(' Want to be a one-dimensional array ')
        param_1 = param_1.reshape((param_1.shape[1],))
        param_2 = param_2.reshape((param_2.shape[1],))
    except IndexError:
        pass
    return param_1, param_2


def accuracy_sco(y_true, y_predict):
    """
    计算准确率
    :param y_true:正确的标签[a one-dimensional array]
    :param y_predict:预测的标签[a one-dimensional array]
    :return:准确率
    """
    y_true, y_predict = check_shape(y_true, y_predict)
    compare_list = (y_predict == y_true).astype(int)  # 转化成bool类型后再转化为int类型
    return np.sum(compare_list) / y_predict.__len__()  # sum统计预测正确的 / 总的标签  =  准确率


def cf_matrix(y_true, y_predict):
    """
    计算混淆矩阵
    :param y_true: 正确的标签[a one-dimensional array]
    :param y_predict: 预测的标签[a one-dimensional array]
    :return:混淆矩阵
    """
    y_true, y_predict = check_shape(y_true, y_predict)
    if np.shape(y_true) != np.shape(y_predict):
        raise ValueError(' Found input variables with inconsistent numbers of samples ')
    type_list = set(y_true)  # 去除重复的标签得出类别的总数
    matrix = []
    for i in type_list:
        type_index = np.where(y_true == i)[0]  # 找到真正标签中属于这一类的样本的标签
        type_pre = y_predict[type_index]  # 找到这些样本的预测结果
        matrix.append([np.sum(type_pre == j) for j in type_list])
        # 寻找原本应该被分到这一类别的样本，预测结果中分到各个类别的个数，并组成一个列表，经过循环后这多组列表组成一个矩阵
    return np.array(matrix)  # 将二维的列表转化为矩阵并返回


def f1_score(y_true, y_predict):
    """
    计算F值
    :param y_true: 正确的标签[a one-dimensional array]
    :param y_predict: 预测的标签[a one-dimensional array]
    :return:F_Score
    """
    y_true, y_predict = check_shape(y_true, y_predict)
    matrix_01_cl = cf_matrix(y_true, y_predict)  # 利用混淆矩阵计算tp，fp，tn，fn
    precision = matrix_01_cl[1][1] / (matrix_01_cl[0][1] + matrix_01_cl[1][1])  # 计算精度 tp / (tp + fp)
    recall = matrix_01_cl[1][1] / np.sum(matrix_01_cl[1])  # 计算召回率  tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)  # 计算F_Score   2*精度*召回率/(精度 + 召回率)
    return f1


def roc_fit(y_sample, y_score, label=None, threshold_num=None):
    """
    绘制ROC曲线
    :param y_sample: 样本标签 (a one-dimensional array)
    :param y_score: 预测模型对样本评定的分数
    :param label: 图像标签(string, default is none)
    :param threshold_num: 划分阈值数量[int, default is none(len of y_score)]
    :return:None
    """
    y_sample, y_score = check_shape(y_sample, y_score)
    if threshold_num is None:
        threshold_num = len(y_score)

    max_score = np.max(y_score)
    min_score = np.min(y_score)
    threshold_list = np.linspace(min_score, max_score, threshold_num)  # 划分多个阈值
    tpr, fpr = [], []

    for i in threshold_list:  # 按每个阈值划分一次预测结果并计算tpr、fpr
        y = (y_score > i).astype(int)
        matrix_y = cf_matrix(y_sample, y)
        tpr.append(matrix_y[1][1] / np.sum(matrix_y[1]))
        fpr.append(matrix_y[0][1] / np.sum(matrix_y[0]))
    tpr, fpr = np.array(tpr), np.array(fpr)  # 将列表转化为array

    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()


def pr_curve(y_sample, y_score, threshold_num=None):
    """
    计算PR曲线
    :param y_sample: 样本标签 (a one-dimensional array)
    :param y_score: 预测模型对样本评定的分数
    :param threshold_num:  划分的阈值数量[int, default is none(len of y_score)]
    :return:None
    """
    y_sample, y_score = check_shape(y_sample, y_score)
    if threshold_num is None:
        threshold_num = len(y_score)

    max_score = np.max(y_score)
    min_score = np.min(y_score)
    threshold_list = np.linspace(min_score, max_score, threshold_num)  # 划分多个阈值
    precisions, recalls = [], []

    for i in threshold_list:  # 按每个阈值划分一次预测结果并计算精度和召回率
        y = (y_score > i).astype(int)
        matrix_01_cl = cf_matrix(y_sample, y)
        precisions.append(matrix_01_cl[1][1] / (matrix_01_cl[0][1] + matrix_01_cl[1][1]))  # 计算精度 tp / (tp + fp)
        recalls.append(matrix_01_cl[1][1] / np.sum(matrix_01_cl[1]))  # 计算召回率  tp / (tp + fn)
    precisions, recalls = np.array(precisions), np.array(recalls)  # 将列表转化为array

    plt.plot(threshold_list, precisions, "b--", label="Precision")
    plt.plot(threshold_list, recalls, "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])


def calculation_auc(y_sample, y_score):
    """
    计算 AUC
    :param y_sample: 样本标签(a one-dimensional array)
    :param y_score: 预测标签(a one-dimensional array)
    :return:AUC (float)
    """
    y_sample, y_score = check_shape(y_sample, y_score)
    positive_index, negative_index = np.where(y_sample == 1)[0], np.where(y_sample == 0)[0]
    positive_score, negative_score = y_score[positive_index], y_score[negative_index]
    score_i = 0
    for i in positive_score.flat:  # 从统计意义上计算，一对正负样品中预测得到正样品概率大于负样品概率的概率
        for j in negative_score.flat:
            if i > j:
                score_i += 1  # 正样品分数大于负样品，也就是预测到正样品的概率更大，加分
            elif i == j:
                score_i += 0.5  # 相同加0.5
    return score_i / (positive_index.shape[0] * negative_index.shape[0])  # 得到的分数/总对数


def mse(y_true, y_predict):
    """
    计算 MSE
    :param y_true:样本标签 (One-dimensional array)
    :param y_predict:预测标签 (One-dimensional array)
    :return:mse (float)
    """
    d_value = y_true - y_predict  # 计算误差
    score = d_value.dot(d_value.T)/y_true.__len__()  # 利用矩阵乘法计算平均平方误差
    return score


def mae(y_true, y_predict):
    """
    计算 MAE
    :param y_true:样本标签 (One-dimensional array)
    :param y_predict:预测标签 (One-dimensional array)
    :return:mae (float)
    """
    return np.linalg.norm(y_true - y_predict, ord=1)/y_true.__len__()  # 利用numpy中计算一范数来计算绝对误差和之后再取平均


def rmse(y_true, y_predict):
    """
    计算 RMSE
    :param y_true:样本标签 (One-dimensional array)
    :param y_predict:预测标签 (One-dimensional array)
    :return:rmse(float)
    """
    d_value = y_true - y_predict  # 计算误差
    score = d_value.dot(d_value) / y_true.__len__()  # 利用矩阵乘法计算平均平方误差
    return np.sqrt(score)  # 均方根误差也就是在平均平方误差的情况下开方


def r_square(y_true, y_predict):
    """
    计算 R平方
    :param y_true:样本标签 (One-dimensional array)
    :param y_predict:预测标签 (One-dimensional array)
    :return:R 平方
    """
    rss_mse = mse(y_true, y_predict)  # mse = rss / 样本个数
    tss_mse = np.var(y_true)  # tss = 方差 / 样本个数
    return 1-rss_mse/tss_mse  # 利用小技巧消去分母，直接利用已有函数计算


def mape(y_true, y_predict):
    """
    计算 MAPE
    :param y_true:样本标签 (One-dimensional array)
    :param y_predict:预测标签 (One-dimensional array)
    :return:mape(float)
    """
    score_abs = np.abs((y_true - y_predict)/y_true)  # 绝对百分比误差
    return np.sum(score_abs*100)/y_true.__len__()  # 取均值，得出平均绝对百分比误差


