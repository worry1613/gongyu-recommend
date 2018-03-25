# /usr/bin/env python
# -*- coding: UTF-8 -*-
# @Time    : 2018/3/23 11:43
# @Author  : houtianba(549145583@qq.com)
# @FileName: itemcf.py
# @Software: PyCharm
# @Blog    ：http://blog.csdn.net/worryabout/

# item协同过滤算法类实现
# 经典版本,进化版本
# 参考网上内容以及《推荐系统实践》书内代码


import math
import pickle
import json
from operator import itemgetter
import numpy as np
import pandas as pd
import datetime
import time
from copy import deepcopy, copy


class itemCF:
    def __init__(self, f):
        self.file = f
        self.df = pd.read_csv(self.file, sep='::', header=None, usecols=[0, 1], names=['userid', 'itemid'],
                              engine='python')
        # alluserids 所有用户的id集合
        self.alluserids = pd.Series(self.df['userid']).unique()
        # allitemids 所有物品的id集合
        self.allitemids = pd.Series(self.df['itemid']).unique()
        self.W = dict()
        self.calOK = False
        # iandiuser  item和item的关系集合
        self.iandiuser = dict()

        try:
            f = open("./data/itemcf.W.dat", "rb")
            self.W = pickle.load(f)
            f.close()
            self.calOK = True
        except Exception as e :
            print('itemcf.W.dat文件不存在')
        try:
            f = open("./data/itemcf.iandiuser.dat", "rb")
            self.iandiuser = pickle.load(f)
            f.close()
            self.calOK = True
        except Exception as e :
            print('itemcf.iandiuser.dat文件不存在')
        # item_users 物品id对应的用户id集合表关系
        self.item_users = dict()
        # user_items 用户id对应的物品id集合表关系
        self.user_items = dict()
        for row in self.df.itertuples():
            self.item_users.setdefault(row[2], set())
            self.item_users[row[2]].add(row[1])
            self.user_items.setdefault(row[1], set())
            self.user_items[row[1]].add(row[2])
        # 每个item的访问用户总量
        self.itemusercount = dict()
        self.itemusercount = self.df.groupby(self.df['itemid']).size().to_dict()

    # t  算法种类
    # 1 -- 传统算法  2 -- 改进算法，性能提高10%-15%
    def fit(self, t=2):
        # 算法分拆成2个函数，方便复用
        if self.calOK is False:
            self.__fit(t)
            self.__fitW()
            self.__normalization()
            try:
                f = open("./data/itemcf.W.dat", "wb")
                pickle.dump(self.W,f)
                f.close()
            except Exception as e:
                print('itemcf.W.dat保存文件出错')
            try:
                f = open("./data/itemcf.iandiuser.dat", "wb")
                pickle.dump(self.iandiuser, f)
                f.close()
            except Exception as e:
                print('itemcf.iandiuser.dat保存文件出错')

    def __fit(self, t):
        start = datetime.datetime.now()
        print('start==', start)
        # ic=0
        if t == 1:
            # 最传统的算法
            for u, items in self.user_items.items():
                for i in items:
                    for j in items:
                        if i == j:
                            continue
                        self.iandiuser.setdefault(i, {})
                        self.iandiuser[i].setdefault(j, 0)
                        self.iandiuser[i][j] += 1
                        # ic+=1
                # print(ic,datetime.datetime.now())
        else:
            # 改进的算法，性能提高10%-15%
            for u, items in self.user_items.items():
                ii = copy(items)
                for i in items:
                    ii.remove(i)
                    for j in ii:
                        # ic += 1
                        self.iandiuser.setdefault(i, {})
                        self.iandiuser[i].setdefault(j, 0)
                        self.iandiuser[i][j] += 1
                        self.iandiuser.setdefault(j, {})
                        self.iandiuser[j].setdefault(i, 0)
                        self.iandiuser[j][i] += 1
                # print(ic,datetime.datetime.now())
        # print('last',ic)
        end = datetime.datetime.now()
        print('end==', end)
        print('times==', end - start)

    def __fitW(self):
        start = datetime.datetime.now()
        print('start==', start)
        for i, ri in self.iandiuser.items():
            for j, cij in ri.items():
                self.W.setdefault(i, {})
                self.W[i].setdefault(j, cij / math.sqrt(self.itemusercount[i] * self.itemusercount[j]))
        end = datetime.datetime.now()
        print('end==', end)
        print('times==', end - start)

    # 数据归一化
    def __normalization(self):
        Wret = self.W.copy()
        for k, v in self.W.items():
            maxW = max([tv for tk, tv in v.items()])
            print(v)
            for kk, vv in v.items():
                self.W[k][kk] = vv / maxW
            print(self.W[k])

    def recommend(self, user, k=10, n=20):
        rank = dict()
        rui = 1.0
        ru = self.user_items[user]
        for i in ru:
            for j, wj in sorted(self.W[i].items(), key=itemgetter(1), reverse=True)[0:k]:
                if j in ru:
                    # we should filter items user interacted before
                    continue
                rank.setdefault(j, {})
                rank[j].setdefault('weight', 0)
                rank[j].setdefault('reason', {})
                rank[j]['reason'].setdefault(i, 0)
                rank[j]['weight'] += wj * rui
                rank[j]['reason'][i] = wj * rui

            ret = sorted(rank.items(), key=lambda d: d[1]['weight'], reverse=True)
            return ret[:n]
        else:
            return []

# item协同过滤进化版，对热门产品进入惩罚
class itemCFIUF(itemCF):
    #
    def __fit(self, t):
        start = datetime.datetime.now()
        print('start==', start)
        # ic=0
        if t == 1:
            # 最传统的算法
            for u, items in self.user_items.items():
                itemc = len(items)
                for i in items:
                    for j in items:
                        if i == j:
                            continue
                        self.iandiuser.setdefault(i, {})
                        self.iandiuser[i].setdefault(j, 0)
                        self.iandiuser[i][j] += 1 / math.log(1 + itemc)
                        # ic+=1
                # print(ic,datetime.datetime.now())
        else:
            # 改进的算法，性能提高10%-15%
            for u, items in self.user_items.items():
                itemc = len(items)
                ii = copy(items)
                for i in items:
                    ii.remove(i)
                    for j in ii:
                        # ic += 1
                        self.iandiuser.setdefault(i, {})
                        self.iandiuser[i].setdefault(j, 0)
                        self.iandiuser[i][j] += 1 / math.log(1 + itemc)
                        self.iandiuser.setdefault(j, {})
                        self.iandiuser[j].setdefault(i, 0)
                        self.iandiuser[j][i] += 1 / math.log(1 + itemc)
                # print(ic,datetime.datetime.now())
        # print('last',ic)
        end = datetime.datetime.now()
        print('end==', end)
        print('times==', end - start)


if __name__ == '__main__':
    itemcf = itemCF('./data/views.dat')
    itemcf.fit()
    r = itemcf.recommend(1)
    r1 = itemcf.recommend(10, k=10, n=5)
    print(r)
    print(r1)

    itemcfiuf = itemCFIUF('./data/views.dat')
    itemcfiuf.fit()
    riif = itemcfiuf.recommend(1)
    r1iif = itemcfiuf.recommend(1, k=20, n=10)
    print(riif)
    print(r1iif)


