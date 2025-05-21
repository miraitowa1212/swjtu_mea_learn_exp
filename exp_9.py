import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk


train_file = 'experiment_09_training_set.csv'
test_file = 'experiment_09_testing_set.csv'


def read_csv(filename: str) -> tuple[np.ndarray, np.ndarray]:
    """
    读取csv文件
    :param filename: csv文件名
    :return: 特征和标签
    """
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    label = data[:, 0]
    feature = data[:, 1:]
    return feature, label


class RandomForest:
    """
    随机森林分类器
    """
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        """
        训练随机森林
        :param X: 特征
        :param y: 标签
        :return: None
        """
        for i in range(self.n_estimators):
            indices = np.random.choice(X.shape[0], int(X.shape[0] * 0.7), replace=True)
            X_sub = X[indices]
            y_sub = y[indices]
            tree = sk.tree.DecisionTreeClassifier(max_depth=self.max_depth, criterion='entropy', max_features=50)
            tree.fit(X_sub, y_sub)
            self.trees.append(tree)

    def predict(self, X):
        """
        预测,通过投票的方式进行预测
        :param X: 特征
        :return: 预测结果
        """
        predictions = np.zeros((X.shape[0], self.n_estimators))
        for i, tree in enumerate(self.trees):
            predictions[:, i] = tree.predict(X)
        predictions = predictions.astype(int)
        return np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=predictions)


    def test(self, X, y):
        """
        测试
        :param X: 特征
        :param y: 标签
        :return: 准确率
        """
        predictions = self.predict(X)
        return np.sum(predictions == y) / y.shape[0]



if __name__ == '__main__':
    X_train, y_train = read_csv(train_file)
    X_test, y_test = read_csv(test_file)
    accs = []
    for i in range(1, 21):
        rf = RandomForest(n_estimators=i, max_depth=7)
        rf.fit(X_train, y_train)
        accs.append(acc := rf.test(X_test, y_test))
        print(f"when n_estimators={i}, acc={acc}")

    plt.plot(accs)
    plt.xlabel('n_estimators')
    plt.ylabel('accuracy')
    plt.title('Random Forest Accuracy')
    plt.show()



