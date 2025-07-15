# -*- coding: utf-8 -*-
# @File    : metrics.py
# @Author  : juntang
# @Time    : 2022/11/17 15:17

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def hit_ratio(out, label):
    out = out.numpy()
    label = label.numpy()
    label_len = len(label)
    hit = 0
    for i in range(label_len):
        if out[i][0] > out[i][1] and label[i] == 0:
            hit += 1
        if out[i][0] <= out[i][1] and label[i] == 1:
            hit += 1
    return hit / label_len


def get_pred(out):
    out_label = []
    for lb in out:
        if lb[0] > lb[1]:
            out_label.append(0)
        else:
            out_label.append(1)
    return np.array(out_label)


def precision(y_true, out):
    return precision_score(y_true.cpu().numpy(), get_pred(out.cpu()), zero_division=0)


def recall(y_true, out):
    return recall_score(y_true.cpu().numpy(), get_pred(out.cpu()), zero_division=0)


def f1(y_true, out):
    return f1_score(y_true.cpu().numpy(), get_pred(out.cpu()), zero_division=0)


def get_pre_result(y_true, out):
    # 返回TP、TN、FP、FN数量，方便后面计算F1-micro
    y_true = y_true.cpu().numpy()
    y_pre = get_pred(out.cpu())
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(len(y_true)):
        if y_true[i] == 1:
            if y_pre[i] == 1:
                TP += 1
            if y_pre[i] == 0:
                FN += 1
        else:
            if y_pre[i] == 1:
                FP += 1
            if y_pre[i] == 0:
                TN += 1
    return TP, TN, FP, FN


def get_score(TP, TN, FP, FN):
    if (TP + FP) == 0 or (TP + FN) == 0 or TP == 0:
        return 0, 0, 0
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


'''
F1-micro 和 F1-macro
F1-micro: 通过不同类别的TP, TN, FP, FN 一起计算
F1-macro: 每种类别的F1 取平均
'''
