import numpy as np
import matplotlib.pyplot as plt

file_train = "experiment_02_training_set.csv"
file_test = "experiment_02_testing_set.csv"


def read_csv(file_name: str) -> list:
    """
    读取CSV文件并返回数据的X和Y列。

    :param file_name: 要读取的CSV文件的名称。
    :return: 包含数据X列和Y列的列表。
    """
    data = np.loadtxt(file_name, delimiter=',', skiprows=1)
    data_x = data[:, 0]
    data_y = data[:, 1]
    return [data_x, data_y]


def mse(list_pred: list, list_true: list) -> float:
    """
        计算预测值列表和真实值列表之间的均方误差 (MSE)。

        :param list_pred: 预测值列表。
        :param list_true: 真实值列表。
        :return: 均方误差 (MSE)。
        """
    assert len(list_pred) == len(list_true), "Length of lists should be equal"
    length = len(list_pred)
    acc_error = 0.0
    for value_true, value_pred in zip(list_true, list_pred):
        acc_error += (value_true - value_pred) ** 2
    return acc_error / length


def draw_scatter_plot(train_data: list, test_data: list, model: tuple, title: str):
    plt.scatter(train_data[0], train_data[1], c='blue', label='Training Set')
    plt.scatter(test_data[0], test_data[1], c='green', label='Testing Set')
    if len(model) == 2:
        x = np.linspace(min(min(train_data[0]), min(test_data[0])),
                        max(max(train_data[0]), max(test_data[0])), 100)
        y = x * model[0] + model[1]
        plt.plot(x, y, c='red', label='Linear Regression')
    elif len(model) == 3:
        x = np.linspace(min(min(train_data[0]), min(test_data[0])),
                        max(max(train_data[0]), max(test_data[0])), 100)
        y = x ** 2 * model[0] + x * model[1] + model[2]
        plt.plot(x, y, c='red', label='Quadratic Regression')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()


class LinearModel:
    def __init__(self):
        self.w_1 = 0.0
        self.w_2 = 0.0

    def fit(self, X: list, y: list):
        n = len(X)
        self.w_1 = ((n * sum(X[i] * y[i] for i in range(n)) - sum(X) * sum(y)) /
                    (n * sum(X[i] ** 2 for i in range(n)) - (sum(X)) ** 2))
        self.w_2 = (sum(y) - self.w_1 * sum(X)) / n

    def predict(self, X: float) -> float:
        return self.w_1 * X + self.w_2


class QuadraticModel:
    def __init__(self):
        self.w_1 = 0.0
        self.w_2 = 0.0
        self.w_3 = 0.0

    def fit(self, X: list, y: list):
        n = len(X)
        Y_sum = sum(y)
        X_sum = sum(X)
        XY_sum = sum(X[i] * y[i] for i in range(n))
        X_2_Y_sum = sum(X[i] ** 2 * y[i] for i in range(n))
        X_2_sum = sum(X[i] ** 2 for i in range(n))
        X_3_sum = sum(X[i] ** 3 for i in range(n))
        X_4_sum = sum(X[i] ** 4 for i in range(n))
        A = np.array([[n, X_sum, X_2_sum],
                      [X_sum, X_2_sum, X_3_sum],
                      [X_2_sum, X_3_sum, X_4_sum]])
        b = [Y_sum, XY_sum, X_2_Y_sum]
        solution = np.linalg.solve(A, b)
        self.w_1 = solution[2]
        self.w_2 = solution[1]
        self.w_3 = solution[0]

    def predict(self, X: float) -> float:
        return self.w_1 * X ** 2 + self.w_2 * X + self.w_3


"""train_data = read_csv(file_train)
test_data = read_csv(file_test)
Linear = LinearModel()
Linear.fit(train_data[0], train_data[1])
train_pred = [Linear.predict(x) for x in test_data[0]]
print(f"Model_w1:{Linear.w_1}, Model_w2:{Linear.w_2}, Train MSE: {mse(train_pred, test_data[1])}")
draw_scatter_plot(train_data, test_data, (Linear.w_1, Linear.w_2), 'Linear Regression')

Quadratic = QuadraticModel()
Quadratic.fit(train_data[0], train_data[1])
train_pred = [Quadratic.predict(x) for x in test_data[0]]
print(f"Model_w1:{Quadratic.w_1}, Model_w2:{Quadratic.w_2}, Model_w3:{Quadratic.w_3}, "
      f"Train MSE: {mse(train_pred, test_data[1])}")
draw_scatter_plot(train_data, test_data, (Quadratic.w_1, Quadratic.w_2, Quadratic.w_3), 'Quadratic Regression')"""
