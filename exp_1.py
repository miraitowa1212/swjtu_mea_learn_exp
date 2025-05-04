import csv
import sys
import numpy as np

file_name_1 = 'experiment_01_dataset_01.csv'
file_name_2 = 'experiment_01_dataset_02.csv'


def mse(list_pred: list, list_true: list) -> float:
    assert len(list_pred) == len(list_true), "Length of lists should be equal"
    length = len(list_pred)
    acc_error = 0.0
    for value_true, value_pred in zip(list_true, list_pred):
        acc_error += (value_true - value_pred) ** 2
    return acc_error / length


def rmse(list_pred: list, list_true: list) -> float:
    return np.sqrt(mse(list_pred, list_true))


def mae(list_pred: list, list_true: list) -> float:
    assert len(list_pred) == len(list_true), "Length of lists should be equal"
    value_max = -sys.maxsize - 1.0
    for value_true, value_pred in zip(list_true, list_pred):
        if abs(value_true - value_pred) > value_max:
            value_max = abs(value_true - value_pred)
    return value_max


def confusion_matrix(list_pred: list, list_true: list) -> list:
    assert len(list_pred) == len(list_true), "Length of lists should be equal"
    true_true = 0
    true_false = 0
    false_true = 0
    false_false = 0
    for value_true, value_pred in zip(list_true, list_pred):
        if value_true == 1.0:
            if value_pred == 1.0:
                true_true += 1
            else:
                true_false += 1
        else:
            if value_pred == 1.0:
                false_true += 1
            else:
                false_false += 1
    matrix = [[true_true, true_false], [false_true, false_false]]
    return matrix


def precision(matrix: list) -> float:
    return matrix[0][0] / (matrix[0][0] + matrix[1][0])


def recall(matrix: list) -> float:
    return matrix[0][0] / (matrix[0][0] + matrix[0][1])


def f1_score(matrix: list) -> float:
    prec = precision(matrix)
    rec = recall(matrix)
    return 2 * (prec * rec) / (prec + rec)


def read_csv(file_name: str) -> list:
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        pred_1 = []
        pred_2 = []
        pred_3 = []
        true_1 = []
        for row in reader:
            true_1.append(float(row[1]))
            pred_1.append(float(row[2]))
            pred_2.append(float(row[3]))
            pred_3.append(float(row[4]))
        data = [true_1, pred_1, pred_2, pred_3]
    return data


"""data_1 = read_csv(file_name_1)

print("Error value")
print("Pred 1:")
print(f"mse:{mse(data_1[1], data_1[0])}")
print(f"mae:{mae(data_1[1], data_1[0])}")
print(f"rmse:{rmse(data_1[1], data_1[0])}")
print('-' * 40)
print("Pred 2:")
print(f"mse:{mse(data_1[2], data_1[0])}")
print(f"mae:{mae(data_1[2], data_1[0])}")
print(f"rmse:{rmse(data_1[2], data_1[0])}")
print('-' * 40)
print("Pred 3:")
print(f"mse:{mse(data_1[3], data_1[0])}")
print(f"mae:{mae(data_1[3], data_1[0])}")
print(f"rmse:{rmse(data_1[3], data_1[0])}")


data_2 = read_csv(file_name_2)
Pred_1_matrix = confusion_matrix(data_2[1], data_2[0])
Pred_2_matrix = confusion_matrix(data_2[2], data_2[0])
Pred_3_matrix = confusion_matrix(data_2[3], data_2[0])
print("\n\nConfusion Matrix:")
print(f"Pred 1:")
print(f"Confusion Matrix: {Pred_1_matrix}")
print(f"Precision: {precision(Pred_1_matrix)}")
print(f"Recall: {recall(Pred_1_matrix)}")
print(f"F1 Score: {f1_score(Pred_1_matrix)}")
print('-' * 40)
print(f"Pred 2:")
print(f"Confusion Matrix: {Pred_2_matrix}")
print(f"Precision: {precision(Pred_2_matrix)}")
print(f"Recall: {recall(Pred_2_matrix)}")
print(f"F1 Score: {f1_score(Pred_2_matrix)}")
print('-' * 40)
print(f"Pred 3:")
print(f"Confusion Matrix: {Pred_3_matrix}")
print(f"Precision: {precision(Pred_3_matrix)}")
print(f"Recall: {recall(Pred_3_matrix)}")
print(f"F1 Score: {f1_score(Pred_3_matrix)}")"""
