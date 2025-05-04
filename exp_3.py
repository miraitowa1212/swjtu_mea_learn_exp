import numpy as np
import matplotlib.pyplot as plt

train_file = 'experiment_03_training_set.csv'
test_file = 'experiment_03_testing_set.csv'


def read_data(file_name: str) -> list:
    data = np.loadtxt(file_name, delimiter=',', skiprows=1)
    X = data[:, :-1]
    y = data[:, -1]
    return [X, y]


def loss_plot(loss_value: list):
    plt.plot(loss_value)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    plt.show()


def confusion_matrix(list_pred: list, list_true: list) -> list:
    assert len(list_pred) == len(list_true), "Length of lists should be equal"
    true_true = 0
    true_false = 0
    false_true = 0
    false_false = 0
    for value_true, value_pred in zip(list_true, list_pred):
        if value_true == 1.0:
            if value_pred == 1.0:
                true_true += 1
            else:
                true_false += 1
        else:
            if value_pred == 1.0:
                false_true += 1
            else:
                false_false += 1
    matrix = [[true_true, true_false], [false_true, false_false]]
    return matrix


def precision(matrix: list) -> float:
    return matrix[0][0] / (matrix[0][0] + matrix[1][0])


def recall(matrix: list) -> float:
    return matrix[0][0] / (matrix[0][0] + matrix[0][1])


def accuracy(matrix: list) -> float:
    return (matrix[0][0] + matrix[1][1]) / (matrix[0][0] + matrix[0][1] + matrix[1][0] + matrix[1][1])


def error_rate(matrix: list) -> float:
    return 1 - accuracy(matrix)


def f1_score(matrix: list) -> float:
    prec = precision(matrix)
    rec = recall(matrix)
    return 2 * (prec * rec) / (prec + rec)


class LogicRegression:
    def __init__(self, learning_rate=0.01, total_iterations=100, threshold=0.5):
        self.learning_rate = learning_rate
        self.total_iterations = total_iterations
        self.threshold = threshold
        self.X = None
        self.y = None
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        训练逻辑回归模型的方法。

        :param X: 特征矩阵，形状为 (n_samples, n_features)
        :param y: 目标向量，形状为 (n_samples,)

        :return: None
        """
        # 保存输入的特征矩阵和目标向量
        self.X = X
        self.y = y
        # 初始化权重向量为零向量，长度等于特征数量
        self.w = np.zeros(X.shape[1])
        # 初始化偏置项为零
        self.b = 0
        # 用于存储每次迭代的平均损失值
        loss_value = []
        # 进行指定次数的迭代训练
        for i in range(self.total_iterations):
            # 初始化总损失为零
            sum_loss = 0
            # 遍历每个样本
            for single_X, single_y in zip(X, y):
                # 计算预测概率
                pred_y = self.sigmoid(np.dot(single_X, self.w) + self.b)
                # 计算当前样本的损失
                loss = -single_y * np.log(pred_y) - (1 - single_y) * np.log(1 - pred_y)
                # 累加总损失
                sum_loss += loss
                # 计算权重的梯度
                d_w = single_X * (pred_y - single_y)
                # 计算偏置项的梯度
                d_b = pred_y - single_y
                # 更新权重
                self.w -= self.learning_rate * d_w
                # 更新偏置项
                self.b -= self.learning_rate * d_b
            # 计算并存储本次迭代的平均损失
            loss_value.append(sum_loss / len(X))
        # 绘制损失随迭代次数变化的曲线
        loss_plot(loss_value)

    def predict(self, X) -> int:
        h = self.sigmoid(np.dot(X, self.w) + self.b)
        if h >= self.threshold:
            return 1
        else:
            return 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


logic_regressor = LogicRegression(total_iterations=100)
train_X, train_y = read_data(train_file)
test_X, test_y = read_data(test_file)
logic_regressor.fit(train_X, train_y)
pred = [logic_regressor.predict(X) for X in test_X]
matrix = confusion_matrix(pred, test_y)
print("Confusion Matrix: ")
print(matrix)
print("Precision: ", precision(matrix))
print("Recall: ", recall(matrix))
print("Accuracy: ", accuracy(matrix))
print("Error Rate: ", error_rate(matrix))
print("F1 Score: ", f1_score(matrix))
