# -*- coding: utf-8 -*-
"""
Feb 24, 2020

@author: Peng
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import regressionmain as reg
import math


# (a)
def mse(target, prediction):
    return (np.square(target - prediction)).mean(axis=None)

# (b)
def cross_val(t, X, K, seed=1, lam=0, train_err=False):
    total = np.shape(X)[0]
    if total == 0:
        print("dataset is empty - abort!")
        return
    np.random.seed(seed)
    shuffled_set = np.copy(X)
    np.random.shuffle(shuffled_set)
    mse_list = []
    ret = []
    for i in range():
        start = math.floor(i * total / K)
        end = math.floor((i + 1) * total / K)
        val_set = shuffled_set[start:end]
        train_set = np.concatenate((shuffled_set[:start], shuffled_set[end:]))
        ret.append(reg.getOLSpoly(train_set, t, k))
    return ret

def best_order(target, predictor, K, seed, ret_err=False, D=None):
    if D==None:
        D = np.shape(target)[0]


def main():
    reg.getdataset("womens100.csv")

if __name__ == "__main__":
    main()