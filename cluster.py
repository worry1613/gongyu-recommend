# -*- coding: utf-8 -*-
# @Time    : 2018/3/24 13:44
# @Author  : houtianba(549145583@qq.com)
# @FileName: cluster.py
# @Software: PyCharm
# @Blog    ：http://blog.csdn.net/worryabout/
# canopy 算法做为k-means算法的预计算，尽快确定K的数量和数值
# k-means算法聚类，确定item所属类别
from collections import defaultdict
from numpy import linalg
import json
import pandas as pd
import math
import random
from numpy import *
import pickle
from datetime import datetime
from copy import deepcopy, copy
from functools import reduce
from sklearn import preprocessing


class Canopy:
    def __init__(self, f, t1=0.8, t2=0.6):
        self.file = f
        self.df = pd.read_csv(self.file, index_col='_id', engine='python')
        # 打开数据文件，数据归一化
        min_max_scaler = preprocessing.MinMaxScaler()
        self.dataset = min_max_scaler.fit_transform(self.df)
        self.t1 = .8
        self.t2 = .6

    def getdata(self):
        return self.dataset

    # 设置初始阈值
    def setThreshold(self, t1, t2):
        if t1 > t2:
            self.t1 = t1
            self.t2 = t2
        else:
            print('t1必须大于t2!')

    # 使用欧式距离进行距离的计算
    def euclideanDistance(self, vec1, vec2):
        return math.sqrt(((vec1 - vec2) ** 2).sum())

    # 根据当前dataset的长度随机选择一个下标
    def getRandIndex(self):
        return random.randint(0, len(self.dataset) - 1)

    def clustering(self):
        if self.t1 == 0:
            print('Please set the threshold.')
        else:
            canopies = []  # 用于存放最终归类结果
            while len(self.dataset) != 0:
                rand_index = self.getRandIndex()
                current_center = self.dataset[rand_index]  # 随机获取一个中心点，定为P点
                current_center_list = []  # 初始化P点的canopy类容器
                delete_list = []  # 初始化P点的删除容器
                self.dataset = delete(
                    self.dataset, rand_index, 0)  # 删除随机选择的中心点P
                for datum_j in range(len(self.dataset)):
                    datum = self.dataset[datum_j]
                    distance = self.euclideanDistance(
                        current_center, datum)  # 计算选取的中心点P到每个点之间的距离
                    if distance < self.t1:
                        # 若距离小于t1，则将点归入P点的canopy类
                        current_center_list.append(datum)
                    if distance < self.t2:
                        delete_list.append(datum_j)  # 若小于t2则归入删除容器
                # 根据删除容器的下标，将元素从数据集中删除
                self.dataset = delete(self.dataset, delete_list, 0)
                canopies.append((current_center, len(current_center_list), current_center_list))
        return canopies


class kMeans:
    def __init__(self, dataset, k, values):
        self.dataset = mat(dataset)
        self.k = k
        self.centroids = mat(values)

    def distEclud(self, vecA, vecB):
        return sqrt(sum(power(vecA - vecB, 2)))  # la.norm(vecA-vecB)

    def randCent(self, dataset, k):
        n = shape(dataset)[1]
        centroids = mat(zeros((k, n)))  # create centroid mat
        for j in range(n):  # create random cluster centers, within bounds of each dimension
            minJ = min(dataset[:, j])
            rangeJ = float(max(dataset[:, j]) - minJ)
            centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))
        return centroids

    def kMeans(self, distMeas=distEclud, createCent=randCent):
        m,n = shape(self.dataset)
        clusterAssment = mat(zeros((m, 2)))  # create mat to assign data points
        # to a centroid, also holds SE of each point
        # self.centroids = createCent(dataset, k)
        clusterChanged = True
        while clusterChanged:
            clusterChanged = False
            for i in range(m):  # for each data point assign it to the closest centroid
                minDist = inf;
                minIndex = -1
                for j in range(self.k):
                    distJI = self.distEclud(self.centroids[j, :], self.dataset[i, :])
                    if distJI < minDist:
                        minDist = distJI;
                        minIndex = j
                if clusterAssment[i, 0] != minIndex: clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist ** 2
            # print(self.centroids)
            for cent in range(self.k):  # recalculate centroids
                ptsInClust = self.dataset[
                    nonzero(clusterAssment[:, 0].A == cent)[0]]  # get all the point in this cluster
                self.centroids[cent, :] = mean(ptsInClust, axis=0)  # assign centroid to mean
        return clusterAssment


if __name__ == '__main__':
    t1 = 1.6
    t2 = .8
    canopys = dict()
    calcanopyOK = False
    gc = Canopy('./data/gongyu.1.csv')
    gc.setThreshold(t1, t2)

    try:
        f = open("./data/cluster.canopys.dat", "rb")
        canopys = pickle.load(f)
        f.close()
        calcanopyOK = True
    except Exception as e:
        print('cluster.canopys.dat文件不存在')
    if calcanopyOK is False:
        g = copy(gc)
        for i in range(0,7):
            gg = copy(g)
            start = datetime.now()
            print('start==', start)
            canopies = gg.clustering()
            end = datetime.now()
            print('end==', end)
            print('times==', end - start)
            canopys[i] = {'c':len(canopies),'ids':[c[0] for c in canopies]}
            print('Get %s initial centers.' % len(canopies))

        def max(x,y):
            if canopys[x]['c'] > canopys[y]['c']:
                return x
            else:
                return y
        def min(x,y):
            if canopys[x]['c'] < canopys[y]['c']:
                return x
            else:
                return y
        maxcanopy = reduce(lambda x,y:max(x,y) ,canopys)
        mincanopy = reduce(lambda x,y:min(x,y) ,canopys)
        canopys.pop(maxcanopy)
        canopys.pop(mincanopy)

        try:
            f = open("./data/cluster.canopys.dat", "wb")
            pickle.dump(canopys, f)
            f.close()
        except Exception as e:
            print('cluster.canopys.dat保存文件出错')

    lencount = reduce(lambda x,y:x+y,[v['c'] for k,v in canopys.items()])
    kcount = lencount//5
    mindistince = inf
    mindi = 0

    for k,v in canopys.items():
        m = abs(v['c']-kcount)
        if m < mindistince:
            mindistince = m
            mindi = k


    ret=None
    try:
        f = open("./data/cluster.kmeans.dat", "rb")
        ret = pickle.load(f)
        f.close()
        calcanopyOK = True
    except Exception as e:
        print('cluster.kmeans.dat文件不存在')

    if ret is None:
        kmns = kMeans(gc.getdata(),canopys[mindi]['c'],canopys[mindi]['ids'])
        start = datetime.now()
        print('start==', start)
        ret = kmns.kMeans()
        end = datetime.now()
        print('end==', end)
        print('times==', end - start)

        try:
            f = open("./data/cluster.kmeans.dat", "wb")
            pickle.dump(ret, f)
            f.close()
        except Exception as e:
            print('cluster.cluster.kmeans.dat保存文件出错')
    # print(ret)
    result = None
    try:
        f = open("./data/cluster.kmeans.result.dat", "rb")
        result = pickle.load(f)
        f.close()
        calcanopyOK = True
    except Exception as e:
        print('cluster.kmeans.result.dat文件不存在')
    if result is None:
        result = defaultdict(set)
        for a,b in zip(ret,gc.getdata()):
            result[int(a[0,0])].add(tuple(b))
            # print(a,b)
        try:
            f = open("./data/cluster.kmeans.result.dat", "wb")
            pickle.dump(result, f)
            f.close()
        except Exception as e:
            print('cluster.cluster.kmeans.result.dat保存文件出错')
    # json.dumps(result,indent=4)
    # print(result)




