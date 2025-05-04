import imageio
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tkinter import filedialog


def plot_loss(loss_list: list):
    plt.plot(loss_list)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    plt.show()


class CovNN(nn.Module):
    def __init__(self):
        super(CovNN, self).__init__()
        self.para_name = "CovNN_3.pth"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(128 * 10 * 10, 128)
        self.fc2 = nn.Linear(128, 10)
        self.criterion = nn.CrossEntropyLoss()
        self.transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
        self.dataset = ImageFolder(root='mnist_png/training', transform=self.transform)
        self.optimizer = None
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 128 * 10 * 10)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def single_train(self, data: torch.Tensor, label: torch.Tensor) -> float:
        """
        执行单批次训练
        :param data: 训练数据张量 (batch_size, 1, H, W)
        :param label: 标签张量 (batch_size,)
        :return: 损失值
        """
        self.optimizer.zero_grad()
        output = self.forward(data)
        loss = self.criterion(output, label)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def m_test(self, user: bool = False):
        if user:
            self.load_state_dict(torch.load(self.para_name))
        test_dataset = ImageFolder(root='mnist_png/testing', transform=self.transform)
        test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            for data, label in test_dataloader:
                data, label = data.to(self.device), label.to(self.device)
                output = self.forward(data)
                _, predicted = torch.max(output.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
        return correct / total

    def m_train(self, batch_size: int = 128, max_epoch: int = 100, early_stop: int = 10):
        """
        执行模型训练
        :param batch_size: 批次大小
        :param max_epoch: 最大训练轮数
        :param early_stop: 早停轮数
        :return: 损失值列表
        """
        try:  # 尝试加载模型参数
            self.load_state_dict(torch.load(self.para_name))
        except FileNotFoundError:
            torch.save(self.state_dict(), self.para_name)

        self.optimizer = optim.Adagrad(self.parameters(), lr=0.01)  # 在模型迁移到设备后再设置优化器
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        losses = []
        max_test_acc = 0
        stop = 0
        for epoch in range(max_epoch):
            sum_loss = 0
            count = 0
            for data, label in dataloader:
                data, label = data.to(self.device), label.to(self.device)
                loss = self.single_train(data, label)
                sum_loss += loss
                count += 1
            test_acc = self.m_test()
            print(f"Epoch [{epoch + 1}/{max_epoch}], Loss: {sum_loss / count:.6f}, Test Accuracy: {test_acc:.6f}")
            losses.append(sum_loss / count)
            if test_acc > max_test_acc:  # 早停
                stop = 0
                max_test_acc = test_acc
                torch.save(self.state_dict(), self.para_name)
            else:
                stop += 1
                if stop > early_stop:
                    print("Early stopping at epoch", epoch + 1)
                    break
        return losses

    def predict(self):
        self.load_state_dict(torch.load(self.para_name))
        filename = filedialog.askopenfilename(title="选择图片", filetypes=[("PNG files", "*.png")])
        image = imageio.v2.imread(filename)
        tensor = torch.Tensor(image).view(-1, 28, 28)
        tensor = tensor.to(self.device)
        output = F.softmax(self.forward(tensor), dim=1)
        poss, predicted = torch.max(output.data, 1)
        print(f"Predicted label: {predicted.item()}, Possibility: {poss.item()}")


def confusion_matrix(net: CovNN, show: bool = False) -> list:
    net.load_state_dict(torch.load(net.para_name))
    test_dataset = ImageFolder(root='mnist_png/testing', transform=net.transform)
    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    confusion = torch.zeros(10, 10)
    with torch.no_grad():
        for data, label in test_dataloader:
            data, label = data.to(net.device), label.to(net.device)
            output = net.forward(data)
            _, predicted = torch.max(output.data, 1)
            for i in range(len(label)):
                confusion[label[i], predicted[i]] += 1
    if show:
        for i in confusion.tolist():
            print(i)
    return confusion.tolist()


def prcision_recall(martix: list):
    precision = []
    recall = []
    sum_tp = 0
    sum_fp = 0
    sum_fn = 0
    for i in range(10):
        tp = 0
        fp = 0
        fn = 0
        for j in range(10):
            if i == j:
                tp += martix[i][j]
            else:
                fp += martix[i][j]
                fn += martix[j][i]
        precision.append(tp / (tp + fp))
        recall.append(tp / (tp + fn))
        sum_tp += tp
        sum_fp += fp
        sum_fn += fn
    macro_precision = sum(precision) / 10
    macro_recall = sum(recall) / 10
    macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall)
    print("Macro Precision:", macro_precision)
    print("Macro Recall:", macro_recall)
    print("Macro F1:", macro_f1)
    micro_precision = sum_tp / (sum_tp + sum_fp)
    micro_recall = sum_tp / (sum_tp + sum_fn)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
    print("Micro Precision:", micro_precision)
    print("Micro Recall:", micro_recall)
    print("Micro F1:", micro_f1)


if __name__ == "__main__":
    my_cnn = CovNN()
    """losses = my_cnn.m_train()
    plot_loss(losses)"""
    # print(my_cnn.m_test())
    # my_cnn.predict()
    ma = confusion_matrix(my_cnn, True)
    prcision_recall(ma)
