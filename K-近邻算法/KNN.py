#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    @File  : KNN.py
    @Author: ChenZhiyuan
    @Date  : 2020/7/10  12:04
    @TODO  : 1. KNN 基本原理的代码实现
             2. 使用k-近邻算法改进约会网站的配对效果
"""
import numpy as np
import operator
from collections import Counter


def creatDataSet():
    """
    创建训练数据集和对应标签
    :return:
    """
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    """
    欧式距离 KNN 分类
    :param inX: 用于分类的输入向量
    :param dataSet: 训练样本集
    :param labels: 训练集对应的标签
    :param k: 最近邻居的数目
    :return:
    """
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # dataSetSize行1列
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)  # sum(0)列相加，sum(1)行相加
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()  # 升序排列后返回索引

    classCount = {}  # 定义一个记录类别次数的字典, 统计k个邻居中，各个标签的个数  {'B':2, 'A':1}
    for i in range(k):
        votelabel = labels[sortedDistIndicies[i]]  # 对应升序距离的标签
        # dict.get(key,default=None),字典的get()方法，返回指定键的值，如果值不在字典中返回default的值
        classCount[votelabel] = classCount.get(votelabel, 0) + 1  # 累加dict某个键对应的值
    # key=operator.itemgetter(1)根据字典的值进行排序
    # key=operator.itemgetter(0)根据字典的键进行排序
    # reverse降序排序字典
    # sorted返回一个列表，如[('B', 2),('A', 1)]
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    Vote = []
    for i in range(k):
        Vote.append(labels[sortedDistIndicies[i]])  # 投票所得第i个邻居的标签
    # Xlabel为一个列表，只有一个元素  [('B', 2)]
    Xlabel = Counter(Vote).most_common(1)  # 求Vote中出现次数最多的元素和出现次数，该元素即为inX的标签

    return sortedClassCount[0][0], Xlabel[0][0]


if __name__ == '__main__':
    #######################
    group, label = creatDataSet()
    # print(group, '\n\n\n', label)

    #######################
    a = classify0([0, 0], group, label, 3)
    print(a[0])

    #######################