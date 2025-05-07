import numpy as np
import scipy as sp

train_file = 'experiment_07_training_set.csv'
test_file = 'experiment_07_testing_set.csv'

def read_csv(filename: str) -> list:
    with open(filename, 'r') as file:
        lines = file.readlines()
        data = []
        for line in lines[1:]:
            values = line.strip().split(',')
            data.append([float(value) for value in values[1:-1]] + [values[-1]])
    return data


def copy_dict(dict1: dict) -> dict:
    dict2 = {}
    for key in dict1.keys():
        dict2[key] = dict1[key]
    return dict2


def cal_mean_std(data: list) -> tuple[np.array, np.array]:
    data = np.array(data)
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    return mean, std


class NaiveBayes:
    """
    朴素贝叶斯分类器类，用于实现朴素贝叶斯分类器的训练和预测。
    该类包含训练方法和预测方法，可根据输入数据和标签进行训练，并对新数据进行预测。
    Attributes:
        data_dict (dict): 用于存储训练数据的字典，键为类别标签，值为对应类别的数据列表。
        mean_var_dict (dict): 用于存储每个类别的均值和方差的字典，键为类别标签，值为对应类别的均值和方差元组。
    """
    def __init__(self):
        self.data_dict = {}
        self.mean_var_dict = {}
        self.prior = {}

    def fit(self, datas):
        """
        训练朴素贝叶斯分类器。
        :param datas: 训练数据，格式为列表，每个元素为一个样本，最后一个元素为样本的类别标签。
        """
        for data in datas:
            if str(data[-1]) not in self.data_dict.keys():
                self.data_dict[data[-1]] = []
            self.data_dict[data[-1]].append(np.array(data[:-1]))
        self.mean_var_dict = copy_dict(self.data_dict)
        for key in self.mean_var_dict.keys():
            self.mean_var_dict[key] = cal_mean_std(self.mean_var_dict[key])
        self.prior = copy_dict(self.mean_var_dict)
        for key in self.prior.keys():
            self.prior[key] = len(self.data_dict[key]) / sum([len(self.data_dict[key]) for key in self.data_dict.keys()])


    def predict(self, data) -> str:
        """
        预测样本的类别标签。
        :param data: 样本特征向量，格式为列表或数组。
        :return: 预测的类别标签。
        """
        data = np.array(data)
        p_dict = copy_dict(self.mean_var_dict)
        for specie in p_dict.keys():
            p = self.prior[specie]
            for i in range(len(data)):
                p *= sp.stats.norm.pdf(data[i], p_dict[specie][0][i], p_dict[specie][1][i])
            p_dict[specie] = p
        return max(p_dict, key=p_dict.get)

    def test(self, datas):
        correct = 0
        for data in datas:
            if self.predict(data[:-1]) == data[-1]:
                correct += 1
        return correct / len(datas)


if __name__ == '__main__':
    na = NaiveBayes()
    data = read_csv(train_file)
    na.fit(data)
    print(na.test(read_csv(test_file)))
    print(f'先验概率：{na.prior}')
    print(f'均值和标准差：{na.mean_var_dict}')