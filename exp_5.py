import numpy as np
import matplotlib.pyplot as plt

train_file = 'experiment_05_training_set.csv'
test_file = 'experiment_05_testing_set.csv'


def read_data(filename: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(filename, delimiter=",", skiprows=1)
    labels = data[:, 0].astype(int)
    features = data[:, 1:] / 255.5
    return features, labels


def to_one_hot(labels: np.ndarray) -> np.ndarray:
    return np.eye(10)[labels]


class LinearLayer:
    """
        线性层类，用于实现神经网络中的线性变换。

        该类包含前向传播和反向传播方法，可计算线性变换的输出以及梯度。

        Attributes:
            weights (np.ndarray): 权重矩阵，形状为 (input_size, output_size)。
            bias (np.ndarray): 偏置向量，形状为 (1, output_size)。
            weights_grad (np.ndarray): 权重的梯度矩阵，形状与 weights 相同。
            bias_grad (np.ndarray): 偏置的梯度向量，形状与 bias 相同。
            inputs (np.ndarray): 前向传播时的输入数据。
            z (np.ndarray): 线性变换的输出结果。
    """

    def __init__(self, input_size: int, output_size: int):
        """
            初始化线性层的参数。

            Args:
                input_size (int): 输入数据的维度。
                output_size (int): 输出数据的维度。
        """
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(1, output_size)
        self.weights_grad = np.zeros_like(self.weights)
        self.bias_grad = np.zeros_like(self.bias)
        self.inputs = None
        self.z = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
            前向传播方法，计算线性变换的输出。

            Args:
                inputs (np.ndarray): 输入数据，形状为 (batch_size, input_size)。

            Returns:
                np.ndarray: 线性变换的输出结果，形状为 (batch_size, output_size)。
        """
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.bias
        return self.z

    def backward(self, grad_a: np.ndarray):
        """
            反向传播方法，计算权重、偏置的梯度以及前一层的梯度。

            Args:
                grad_a (np.ndarray): 来自下一层的梯度，形状为 (batch_size, output_size)。

            Returns:
                np.ndarray: 前一层的梯度，形状为 (batch_size, input_size)。
        """
        self.weights_grad = np.dot(self.inputs.T, grad_a) / len(grad_a)
        self.bias_grad = np.mean(grad_a, axis=0, keepdims=True)
        grad_prev = np.dot(grad_a, self.weights.T)
        return grad_prev


class SigmoidLayer:
    def __init__(self):
        self.a = None

    def forward(self, z: np.ndarray) -> np.ndarray:
        self.a = 1 / (1 + np.exp(-z))
        return self.a

    def backward(self, grad_next: np.ndarray) -> np.ndarray:
        grad = grad_next * self.a * (1 - self.a)
        return grad


def mse(inputs: np.ndarray, targets: np.ndarray) -> float:
    loss = np.mean((inputs - targets) ** 2)
    return loss


class Model:
    def __init__(self):
        self.a2 = None
        self.z2 = None
        self.a1 = None
        self.z1 = None
        self.layer1 = LinearLayer(784, 12)
        self.sigmoid1 = SigmoidLayer()
        self.layer2 = LinearLayer(12, 10)
        self.sigmoid2 = SigmoidLayer()

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
            前向传播方法，计算模型的输出。

            Args:
                inputs (np.ndarray): 输入数据，形状为 (batch_size, 784)。

            Returns:
                np.ndarray: 模型的输出，形状为 (batch_size, 10)。
        """
        self.z1 = self.layer1.forward(inputs)
        self.a1 = self.sigmoid1.forward(self.z1)
        self.z2 = self.layer2.forward(self.a1)
        self.a2 = self.sigmoid2.forward(self.z2)
        return self.a2

    def backward(self, labels: np.ndarray):
        sita2 = 2 * (self.sigmoid2.a - labels)
        sita2 = self.sigmoid2.backward(sita2)

        grad_next = self.layer2.backward(sita2)
        sita1 = self.sigmoid1.backward(grad_next)
        self.layer1.backward(sita1)

    def train(self, train_data: np.ndarray, train_labels: np.ndarray,
              lr: float = 0.001, batch_size: int = 100, epochs: int = 2000):
        """
            训练模型的方法，使用 Mini-Batch 梯度下降法进行训练。

            Args:
                train_data (np.ndarray): 训练数据，形状为 (num_samples, 784)。
                train_labels (np.ndarray): 训练标签，形状为 (num_samples,)。
                lr (float, optional): 学习率，控制参数更新的步长。默认为 0.001。
                batch_size (int, optional): 每个 Mini-Batch 的样本数量。默认为 100。
                epochs (int, optional): 训练的轮数。默认为 2000。
        """
        train_labels = to_one_hot(train_labels)
        num_samples = train_data.shape[0]
        losses = []

        for epoch in range(epochs):
            # Mini-Batch 训练
            sum_loss = 0
            permutation = np.random.permutation(num_samples)
            for i in range(0, num_samples, batch_size):
                indices = permutation[i:i + batch_size]
                X_batch = train_data[indices]
                y_batch = train_labels[indices]

                # 前向传播
                outputs = self.forward(X_batch)
                loss = mse(outputs, y_batch)
                sum_loss += loss * X_batch.shape[0]

                # 反向传播
                self.backward(y_batch)

                # 参数更新
                self.layer1.weights -= lr * self.layer1.weights_grad
                self.layer1.bias -= lr * self.layer1.bias_grad
                self.layer2.weights -= lr * self.layer2.weights_grad
                self.layer2.bias -= lr * self.layer2.bias_grad

            losses.append(sum_loss / num_samples)
            print(f"Epoch {epoch + 1}, Loss: {sum_loss / num_samples}")
        plt.plot(losses)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()

    def test(self, inputs: np.ndarray, labels: np.ndarray) -> float:
        accuracy = 0
        for single_input, label in zip(inputs, labels):
            output = np.argmax(self.forward(single_input), axis=1)
            if output == label:
                accuracy += 1
        return accuracy / len(inputs)


my_model = Model()
train_features, train_labels = read_data(train_file)
test_features, test_labels = read_data(test_file)
my_model.train(train_features, train_labels, lr=0.01, batch_size=30, epochs=1000)
accuracy = my_model.test(test_features, test_labels)
print(f"Accuracy on testing set: {accuracy}")
