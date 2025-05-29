import numpy as np
import matplotlib.pyplot as plt


file_name = 'experiment_10_training_set.csv'

def read_csv(filename: str) -> np.ndarray:
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    return data


class KMeans:
    """K-Means clustering algorithm implementation.

        Attributes:
            k: Number of clusters
            max_iter: Maximum iterations for convergence
            centroids: Array of cluster centroids (shape: [k, features])
        """
    def __init__(self, k=2, max_iter=100):
        self.k = k
        self.max_iter = max_iter
        self.centroids = None

    def fit(self, data):
        """
        训练KMeans模型
        :param data: 训练数据
        :return: 损失值
        """

        # 随机初始化质心
        indices = np.random.choice(data.shape[0], self.k, replace=False)
        self.centroids = data[indices]

        for _ in range(self.max_iter):
            # 计算每个点到质心的距离
            distances = np.sqrt(((data - self.centroids[:, np.newaxis])**2).sum(axis=2))
            # 分配每个点到最近的质心
            labels = np.argmin(distances, axis=0)
            # 更新质心
            new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(self.k)])
            # 如果质心没有变化，则停止迭代
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

        loss = 0
        for i in range(self.k):
            loss += np.sum((data[labels == i] - self.centroids[i])**2)

        plt.scatter(data[:, 0], data[:, 1], c=labels)
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', marker='x')
        plt.title(f'KMeans Clustering (K={self.k})')
        plt.show()
        return loss


if __name__ == '__main__':
    data = read_csv(file_name)
    losses = []
    for k in range(1, 11):
        kmeans = KMeans(k=k)
        losses.append(loss := kmeans.fit(data))
        print(f'K={k}, loss={loss}')

    plt.plot(range(1, 11), losses)
    plt.xlabel('K')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    plt.show()




