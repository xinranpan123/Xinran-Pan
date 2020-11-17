#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
==========================================
Copyright (C) 2020 Xinran Pan
All rights reserved
Description:
Created by Xinran Pan at 2020/11/16 8:15
Email:xinranpan@hotmail.com
==========================================
"""

import pandas as pd
import numpy as np
import math
from numpy import array


# 定义熵值法函数
def cal_weight(x):
    '''熵值法计算变量的权重'''
    # 标准化
    # minmax_scaler = lambda x: ((x - np.min(x)) / (np.max(x) - np.min(x)))
    # x = minmax_scaler(x)
    x = x.apply(lambda x: ((x - np.min(x)) / (np.max(x) - np.min(x))))

    # 求k
    rows = x.index.size  # 行
    cols = x.columns.size  # 列
    k = 1.0 / math.log(rows)

    lnf = [[None] * cols for i in range(rows)]

    # 矩阵计算--
    # 信息熵
    # p=array(p)
    x = array(x)
    lnf = [[None] * cols for i in range(rows)]
    lnf = array(lnf)
    for i in range(0, rows):
        for j in range(0, cols):
            if x[i][j] == 0:
                lnfij = 0.0
            else:
                p = x[i][j] / x.sum(axis=0)[j]
                lnfij = math.log(p) * p * (-k)
            lnf[i][j] = lnfij
    lnf = pd.DataFrame(lnf)
    E = lnf

    # 计算冗余度
    d = 1 - E.sum(axis=0)
    # 计算各指标的权重
    w = [[None] * 1 for i in range(cols)]
    for j in range(0, cols):
        wj = d[j] / sum(d)
        w[j] = wj
        # 计算各样本的综合得分,用最原始的数据

    w = pd.DataFrame(w)
    return w


if __name__ == '__main__':
    # 1读取数据
    import pandas as pd
    # df = np.loadtxt('OriginalData.txt')
    df = pd.read_csv('ReplacedData.txt', sep=' ', header=None)
    # df = np.loadtxt('ReplacedData.txt')
    # 2数据预处理 ,去除空值的记录
    # df.dropna()
    # 计算df各字段的权重
    w = cal_weight(df)  # 调用cal_weight
    # w.index = df.columns
    # w.columns = ['weight']
    print(w)
    np.savetxt('entropy_vector.txt',w)

    # 依据熵的权重对每个工作进行打分
    scores = []
    for i in range(len(df)):
        scores.append(np.dot(df.values[i], w.values)[0])
    # 每个人对8
    work_choice = []
    for i in range(int(len(scores)/8)):
        work_scores = scores[i*8:(i+1)*8]
        max_score_index = work_scores.index(max(work_scores))
        print(work_scores, max_score_index)
        work_choice.append(max_score_index)
    np.savetxt('work_choice.txt', work_choice)
    print('love world')
    print('运行完成!')

