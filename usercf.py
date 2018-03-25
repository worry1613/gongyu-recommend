# /usr/bin/env python
# -*- coding: UTF-8 -*-
# @Time    : 2018/3/23 11:43
# @Author  : houtianba(549145583@qq.com)
# @FileName: usercf.py
# @Software: PyCharm
# @Blog    ：http://blog.csdn.net/worryabout/

# 用户协同过滤算法类实现
# 经典版本,进化版本
# 参考网上内容以及《推荐系统实践》书内代码


import math
import json
import pickle
from operator import itemgetter
import numpy as np
import pandas as pd
import datetime
import time
from copy import deepcopy, copy

class userCF:
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
        # uanduitem  用户和用户的关系集合
        self.uanduitem = dict()
        try:
            f = open("./data/usercf.W.dat", "rb")
            self.W = pickle.load(f)
            f.close()
            self.calOK = True
        except Exception as e :
            print('usercf.W.dat文件不存在')
        try:
            f = open("./data/usercf.uanduitem.dat", "rb")
            self.uanduitem = pickle.load(f)
            f.close()
            self.calOK = True
        except Exception as e :
            print('usercf.uanduitem.dat文件不存在')

        # item_users 物品id对应的用户id集合表关系
        self.item_users = dict()
        # user_items 用户id对应的物品id集合表关系
        self.user_items = dict()

        for row in self.df.itertuples():
            self.item_users.setdefault(row[2], set())
            self.item_users[row[2]].add(row[1])
            self.user_items.setdefault(row[1], set())
            self.user_items[row[1]].add(row[2])
        # 每个用户的访问总量
        self.useritemcount = dict()
        self.useritemcount = self.df.groupby(self.df['userid']).size().to_dict()

    # t  算法种类
    # 1 -- 传统算法  2 -- 改进算法，性能提高10%-15%
    def fit(self, t=2):
        # 算法分拆成2个函数，方便复用
        if self.calOK is False:
            self._fit(t)
            self._fitW()
            try:
                f = open("./data/usercf.W.dat", "wb")
                pickle.dump(self.W,f)
                f.close()
            except Exception as e:
                print('usercf.W.dat保存文件出错')
            try:
                f = open("./data/usercf.uanduitem.dat", "wb")
                pickle.dump(self.uanduitem, f)
                f.close()
            except Exception as e:
                print('usercf.uanduitem.dat保存文件出错')

    def _fit(self, t):
        start = datetime.datetime.now()
        print('start==', start)
        # ic=0
        if t == 1:
            # 最传统的算法
            for i, users in self.item_users.items():
                for u in users:
                    for v in users:
                        if u == v:
                            continue
                        self.uanduitem.setdefault(u, {})
                        self.uanduitem[u].setdefault(v, 0)
                        self.uanduitem[u][v] += 1
                        # ic+=1
                # print(ic,datetime.datetime.now())
        else:
            # 改进的算法，性能提高10%-15%
            for i, users in self.item_users.items():
                vs = copy(users)
                for u in users:
                    vs.remove(u)
                    for v in vs:
                        # ic += 1
                        self.uanduitem.setdefault(u, {})
                        self.uanduitem[u].setdefault(v, 0)
                        self.uanduitem[u][v] += 1
                        self.uanduitem.setdefault(v, {})
                        self.uanduitem[v].setdefault(u, 0)
                        self.uanduitem[v][u] += 1
                # print(ic,datetime.datetime.now())
        # print('last',ic)
        end = datetime.datetime.now()
        print('end==', end)
        print('times==', end - start)

    def _fitW(self):
        start = datetime.datetime.now()
        print('start==', start)
        for u, ru in self.uanduitem.items():
            for v, cuv in ru.items():
                self.W.setdefault(u, {})
                self.W[u].setdefault(v, cuv / math.sqrt(self.useritemcount[u] * self.useritemcount[v]))
        end = datetime.datetime.now()
        print('end==', end)
        print('times==', end - start)

    def recommend(self, user, k=10, n=20):
        rank = dict()
        rvi = 1.0
        interacted_items = self.user_items[user]
        if user in self.W.keys():
            for v, wuv in sorted(self.W[user].items(), key=itemgetter(1), reverse=True)[0:k]:
                for i in self.user_items[v]:
                    if i in interacted_items:
                        # we should filter items user interacted before
                        continue
                    rank.setdefault(i, 0)
                    rank[i] += wuv * rvi

            ret = sorted(rank.items(), key=itemgetter(1), reverse=True)
            return ret[:n]
        else:
            return []


# 用户协同过滤进化版，对热门产品进入惩罚
class userCFIIF(userCF):
    #
    def _fit(self, t):
        start = datetime.datetime.now()
        print('start==', start)
        # ic=0
        if t == 1:
            # 最传统的算法
            for i, users in self.item_users.items():
                userc = len(users)
                for u in users:
                    for v in users:
                        if u == v:
                            continue
                        self.uanduitem.setdefault(u, {})
                        self.uanduitem[u].setdefault(v, 0)
                        self.uanduitem[u][v] += 1 / math.log(1 + userc)
                        # ic+=1
                # print(ic,datetime.datetime.now())
        else:
            # 改进的算法，性能提高10%-15%
            for i, users in self.item_users.items():
                userc = len(users)
                vs = copy(users)
                for u in users:
                    vs.remove(u)
                    for v in vs:
                        # ic += 1
                        self.uanduitem.setdefault(u, {})
                        self.uanduitem[u].setdefault(v, 0)
                        self.uanduitem[u][v] += 1 / math.log(1 + userc)
                        self.uanduitem.setdefault(v, {})
                        self.uanduitem[v].setdefault(u, 0)
                        self.uanduitem[v][u] += 1 / math.log(1 + userc)
                # print(ic,datetime.datetime.now())
        # print('last',ic)
        end = datetime.datetime.now()
        print('end==', end)
        print('times==', end - start)





if __name__ == '__main__':
    ucf = userCF('./data/views.dat')
    ucf.fit()
    r = ucf.recommend(1)
    r1 = ucf.recommend(1, k=20, n=10)
    print(r)
    print(r1)

    ucfiif = userCFIIF('./data/views.dat')
    ucfiif.fit()
    riif = ucfiif.recommend(1)
    r1iif = ucfiif.recommend(1, k=20, n=10)
    print(riif)
    print(r1iif)

    ucftime = userCFTIME('./data/views.dat')
    ucftime.fit(alpha=0.2)
    rt = ucftime.recommend(1)
    r1t = ucftime.recommend(1, k=20, n=10)
    print(rt)
    print(r1t)
