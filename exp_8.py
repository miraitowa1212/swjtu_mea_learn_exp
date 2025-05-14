import numpy as np
import matplotlib.pyplot as plt


def cal_entropy_with_weight(data: np.ndarray) -> float:
    """计算带权信息熵
        Args:
            data: 包含特征和权重的数组，data[:, -2]是类别标签，data[:, -1]是样本权重
        Returns:
            加权后的信息熵值
        """
    classes = np.unique(data[:, -2])
    total_weight = np.sum(data[:, -1])
    if total_weight == 0:
        return 0.0
    entropy = 0.0
    for c in classes:
        weight_c = np.sum(data[data[:, -2] == c][:, -1])
        p_c = weight_c / total_weight
        if p_c > 0:
            entropy += -p_c * np.log2(p_c)
    return entropy

class DecisionTree:
    """

    Attributes:
        max_depth (int): 树的最大深度
        tree (dict): 决策树的结构

    """
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, data: np.ndarray, weights: np.ndarray):
        """
        训练决策树
        Args:
            data: 训练数据，包含特征和标签，最后一列是标签
            weights: 样本权重
        """
        """data: [features..., label], weights: 1D array"""
        self.tree = self._build_tree(data, weights, depth=0)

    def _build_tree(self, data, weights, depth):
        """
        递归构建决策树
        :param data: 训练数据，包含特征和标签，最后一列是标签
        :param weights: 样本权重
        :param depth: 当前深度
        :return: 决策树节点
        """
        if depth >= self.max_depth or self._is_leaf(data, weights):
            return self._create_leaf(data, weights)

        feature, threshold = self._find_best_split(data, weights)
        left_mask = data[:, feature] <= threshold
        right_mask = ~left_mask

        left = self._build_tree(data[left_mask], weights[left_mask], depth + 1)
        right = self._build_tree(data[right_mask], weights[right_mask], depth + 1)

        return {
            'feature': feature,
            'threshold': threshold,
            'left': left,
            'right': right
        }

    def _is_leaf(self, data, weights):
        """停止分裂的条件：达到最大深度，或所有样本权重集中于一类
        Args:
            data: 训练数据
            weights: 样本权重
        Returns:
            bool: 是否为叶节点
        """
        labels = data[:, -1]
        unique_labels = np.unique(labels)
        if len(unique_labels) == 1:
            return True

        total_weight = np.sum(weights)
        for label in unique_labels:
            if np.sum(weights[labels == label]) / total_weight > 0.99:
                return True
        return False

    def _create_leaf(self, data, weights):
        """创建叶节点，返回预测类别
        Args:
            data: 训练数据
            weights: 样本权重
        Returns:
            dict: 叶节点信息

        """
        labels = data[:, -1]
        weighted_counts = {}
        for label in np.unique(labels):
            weighted_counts[label] = np.sum(weights[labels == label])
        best_label = max(weighted_counts, key=weighted_counts.get)
        return {'label': best_label}

    def _find_best_split(self, data, weights):
        """
        找到最佳分割特征和阈值
        :param data: 训练数据
        :param weights: 样本权重
        :return: 最佳分割特征和阈值
        """
        n_features = data.shape[1] - 1
        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        for feature in range(n_features):
            sorted_indices = np.argsort(data[:, feature])
            sorted_data = data[sorted_indices]
            sorted_weights = weights[sorted_indices]
            thresholds = (sorted_data[:-1, feature] + sorted_data[1:, feature]) / 2

            for threshold in thresholds:
                left = sorted_data[sorted_data[:, feature] <= threshold]
                left_weights = sorted_weights[sorted_data[:, feature] <= threshold]
                right = sorted_data[sorted_data[:, feature] > threshold]
                right_weights = sorted_weights[sorted_data[:, feature] > threshold]

                gain = self._information_gain(left, left_weights, right, right_weights)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, left, left_weights, right, right_weights):
        """
        计算信息增益
        :param left: 左子树数据
        :param left_weights: 左子树样本权重
        :param right: 右子树数据
        :param right_weights: 右子树样本权重
        :return: 信息增益值
        """
        total_weight = np.sum(left_weights) + np.sum(right_weights)
        entropy_left = cal_entropy_with_weight(np.hstack((left, left_weights.reshape(-1, 1))))
        entropy_right = cal_entropy_with_weight(np.hstack((right, right_weights.reshape(-1, 1))))
        weighted_entropy = (np.sum(left_weights) / total_weight) * entropy_left + \
                           (np.sum(right_weights) / total_weight) * entropy_right

        return cal_entropy_with_weight(np.hstack((np.concatenate((left, right), axis=0),
                                                  (np.concatenate((left_weights, right_weights), axis=0)).reshape(-1, 1)))) - weighted_entropy

    def predict(self, data):
        return np.array([self._predict_one(x, self.tree) for x in data])

    def _predict_one(self, x, node):
        if 'label' in node:
            return node['label']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        else:
            return self._predict_one(x, node['right'])


class AdaBoost:
    """
    AdaBoost算法实现
    Attributes:
        max_weak_learn (int): 最大弱学习器数量
        weak_learners (list): 弱学习器列表
        alphas (list): 弱学习器权重列表
        weights (np.ndarray): 样本权重
    """
    def __init__(self, max_weak_learn: int = 10):
        self.max_weak_learn = max_weak_learn
        self.weak_learners = []
        self.alphas = []
        self.weights = None

    def fit(self, data):
        """
        训练AdaBoost模型
        :param data: 训练数据，包含特征和标签，最后一列是标签
        :return: None
        """
        n_samples = data.shape[0]
        weights = np.ones(n_samples) / n_samples
        for i in range(self.max_weak_learn):
            print(f"Training weak learner {i+1}/{self.max_weak_learn}")
            weak_learner = DecisionTree(max_depth=3)
            weak_learner.fit(data, weights)
            predictions = weak_learner.predict(data)
            error = sum(weights[i] for i in range(n_samples) if predictions[i] != data[i][-1])
            alpha = 0.5 * np.log((1 - error + 1e-10) / (error + 1e-10))
            weights *= np.exp(-alpha * data[:, -1] * predictions)
            weights /= np.sum(weights)
            self.weak_learners.append(weak_learner)
            self.alphas.append(alpha)
        self.weights = weights

    def predict(self, data):
        predictions = np.zeros(data.shape[0])
        for i in range(self.max_weak_learn):
            p = self.weak_learners[i].predict(data) * self.alphas[i]
            predictions += p
        return np.sign(predictions)

    def test(self, data):
        predictions = self.predict(data)
        return sum(predictions[i] == data[i][-1] for i in range(data.shape[0])) / data.shape[0]


def read_csv(filename: str) -> np.ndarray:
    data = np.loadtxt(filename, delimiter=',', skiprows=1)

    return data

if __name__ == '__main__':
    train_file = 'experiment_08_training_set.csv'
    test_file = 'experiment_08_testing_set.csv'
    train_data = read_csv(train_file)
    test_data = read_csv(test_file)
    accs = []

    for i in range(1, 11):
        ada = AdaBoost(max_weak_learn=i)

        ada.fit(train_data)
        accs.append(ada.test(test_data))
    plt.plot(accs)
    print(accs)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Plot')
    plt.show()


