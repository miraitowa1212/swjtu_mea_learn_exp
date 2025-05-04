import numpy as np
import matplotlib.pyplot as plt

train_file = 'experiment_06_training_set.csv'
test_file = 'experiment_06_testing_set.csv'


def read_csv(filename: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    data_x = data[:, 0:2]
    data_y = data[:, -1]
    return data_x, data_y


def plot_data(data_x: np.ndarray, data_y: np.ndarray, w: np.ndarray, b: float):
    class1 = data_x[data_y == 1]
    class2 = data_x[data_y == -1]
    plt.scatter(class1[:, 0], class1[:, 1], c='r', label='Class 1')
    plt.scatter(class2[:, 0], class2[:, 1], c='b', label='Class 2')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Data Distribution')
    plt.legend()
    x = np.linspace(0, 4, 100)
    y = (-w[0] * x - b) / w[1]
    plt.plot(x, y)
    plt.show()


def plot_accs(accs: list):
    plt.plot(accs)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Plot')
    plt.show()


class SoftMarginSVM:
    """
        软间隔支持向量机类，用于实现软间隔支持向量机的训练和预测。
        该类包含训练方法和预测方法，可根据输入数据和标签进行训练，并对新数据进行预测。
        Attributes:
            w (np.ndarray): 权重向量，形状为 (2,)。
            b (float): 偏置项。
            c (float): 软间隔参数，用于控制间隔边界的宽度。
    """

    def __init__(self, c=100):
        self.w = np.array([0.0, 0.0])
        self.b = 0.0
        self.c = c

    def train(self, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray,
              learning_rate: float, epochs: int):
        n_samples = train_x.shape[0]
        accs = []
        for epoch in range(epochs):
            random_index = np.random.randint(n_samples)  # 随机选择一个训练样本的索引
            x_i = train_x[random_index]
            y_i = train_y[random_index]
            if y_i * (np.dot(self.w, x_i) + self.b) >= 1:  # 判断样本是否正确分类且在间隔边界外
                self.w -= learning_rate * self.w
            else:
                self.w -= learning_rate * (self.w - self.c * y_i * x_i)
                self.b -= learning_rate * (-self.c * y_i)
            accs.append(self.test(test_x, test_y))  # 计算当前模型在测试集上的准确率并添加到列表中
        return self.w, self.b, accs

    def predict(self, data_x: np.ndarray) -> np.ndarray:
        return np.sign(np.dot(data_x, self.w) + self.b)

    def test(self, data_x: np.ndarray, data_y: np.ndarray) -> float:
        predictions = self.predict(data_x)
        correct = np.sum(predictions == data_y)
        return correct / len(data_y)


train_x, train_y = read_csv(train_file)
test_x, test_y = read_csv(test_file)
svm = SoftMarginSVM()
w, b, accs = svm.train(train_x, train_y, test_x, test_y, 0.001, 100)
plot_data(test_x, test_y, w, b)
plot_accs(accs)
print(w, b)
print(accs[-1])
