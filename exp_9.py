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


if __name__ == '__main__':
    train_feature, train_label = read_csv(train_file)
    test_feature, test_label = read_csv(test_file)
    accs = []
    for i in range(1, 21):  # 创建随机森林分类器（决策树数量i，最大深度6，每树最大特征数50，使用信息熵）
        random_forest = sk.ensemble.RandomForestClassifier(n_estimators=i, max_depth=6, max_features=50,
                                                           criterion="entropy")
        random_forest.fit(train_feature, train_label)
        accs.append(acc := random_forest.score(test_feature, test_label))  # 记录测试集准确率
        print(f'When n_estimators={i}, the accuracy is {acc}')
    # 绘制准确率变化曲线
    plt.plot(range(1, 21), accs)
    plt.xlabel('n_estimators')
    plt.ylabel('accuracy')
    plt.title('Random Forest')
    plt.show()
