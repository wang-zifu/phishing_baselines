import numpy as np


def calc_values(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    TP = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    TN = np.sum(np.logical_and(y_true == 0, y_pred == 0))
    FP = np.sum(np.logical_and(y_true == 0, y_pred == 1))
    FN = np.sum(np.logical_and(y_true == 1, y_pred == 0))
    return TP, TN, FP, FN


def precision(y_true, y_pred):
    TP, TN, FP, FN = calc_values(y_true, y_pred)
    return TP/(TP + FP)


def recall(y_true, y_pred):
    TP, TN, FP, FN = calc_values(y_true, y_pred)
    return TP/(TP + FN)


def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2*((prec*rec) / (prec + rec))


def false_positive_rate(y_true, y_pred):
    TP, TN, FP, FN = calc_values(y_true, y_pred)
    return FP/(FP + TN)
