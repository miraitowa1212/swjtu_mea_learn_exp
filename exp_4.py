import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

train_file = 'experiment_04_training_set.csv'
test_file = 'experiment_04_testing_set.csv'


def read_data(filename: str) -> list[np.ndarray]:
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    x = data[:, :-1]
    y = data[:, -1].astype(int)
    return [x, y]


train_x, train_y = read_data(train_file)
test_x, test_y = read_data(test_file)
criterions = ['gini', 'entropy']
# 遍历不同的决策树分裂准则
for criterion in criterions:
    # 遍历不同的树最大深度，范围从 1 到 3
    for max_depth in range(1, 4):
        # 创建一个决策树分类器实例，设置随机种子为 1，指定分裂准则和最大深度
        decision_tree = DecisionTreeClassifier(random_state=1, criterion=criterion, max_depth=max_depth)
        # 使用训练数据对决策树分类器进行训练
        decision_tree.fit(train_x, train_y)
        # 计算模型在测试数据上的准确率
        test_accuracy = decision_tree.score(test_x, test_y)
        # 打印当前分裂准则、最大深度和测试准确率
        print(f"Criterion: {criterion}, Max Depth: {max_depth}, Test Accuracy: {test_accuracy}")
        # 绘制决策树，指定特征名称、类别名称，并填充节点颜色
        plot_tree(decision_tree, feature_names=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13'],
                  class_names=['0', '1', '2'], filled=True)
        # 显示绘制好的决策树图形
        plt.show()

