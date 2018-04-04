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
import random

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
        # item_users 物品id对应的用户id集合表关系
        self.item_users = dict()
        # user_items 用户id对应的物品id集合表关系
        self.user_items = dict()
        # for row in self.df.itertuples():
        #     self.item_users.setdefault(row[2], set())
        #     self.item_users[row[2]].add(row[1])
        #     self.user_items.setdefault(row[1], set())
        #     self.user_items[row[1]].add(row[2])
        # 每个item的访问用户总量
        self.itemusercount = dict()
        # self.itemusercount = self.df.groupby(self.df['itemid']).size().to_dict()
        self.test = dict()
        self.train = dict()

        try:
            f = open("./data/itemcf.train.dat", "rb")
            self.train = pickle.load(f)
            f.close()
            self.calOK = True
            self.user_items = deepcopy(self.train)
        except Exception as e:
            self.calOK = False
            print('itemcf.train.dat文件不存在')

        try:
            f = open("./data/itemcf.test.dat", "rb")
            self.test = pickle.load(f)
            f.close()
            self.calOK = True
        except Exception as e:
            self.calOK = False
            print('itemcf.test.dat文件不存在')
        try:
            f = open("./data/itemcf.item_users.dat", "rb")
            self.item_users = pickle.load(f)
            f.close()
            self.calOK = True
        except Exception as e:
            self.calOK = False
            print('itemcf.item_users.dat文件不存在')
        try:
            f = open("./data/itemcf.itemusercount.dat", "rb")
            self.itemusercount = pickle.load(f)
            f.close()
            self.calOK = True
        except Exception as e:
            self.calOK = False
            print('itemcf.itemusercount.dat文件不存在')



    def splitdata(self, M, key):
        """把数据切成训练集和测试集
        :param M:  数据将分成M份
        :param key:  选取第key份数据做为测试数据
        :return:
        """
        if self.calOK is False:
            random.seed(int(time.time()))
            for row in self.df.itertuples():
                if random.randint(0, M) == key:
                    self.test.setdefault(row[1], set())
                    self.test[row[1]].add(row[2])
                else:
                    self.train.setdefault(row[1], set())
                    self.train[row[1]].add(row[2])
                    self.item_users.setdefault(row[2], set())
                    self.item_users[row[2]].add(row[1])

            for k, v in self.item_users.items():
                self.itemusercount.setdefault(k, len(v))
            self.user_items = deepcopy(self.train)

            try:
                f = open("./data/itemcf.train.dat", "wb")
                pickle.dump(self.train, f)
                f.close()
            except Exception as e:
                print('itemcf.train.dat保存文件出错')

            try:
                f = open("./data/itemcf.test.dat", "wb")
                pickle.dump(self.test, f)
                f.close()
            except Exception as e:
                print('itemcf.test.dat保存文件出错')
            try:
                f = open("./data/itemcf.item_users.dat", "wb")
                pickle.dump(self.item_users, f)
                f.close()
            except Exception as e:
                print('itemcf.item_users.dat保存文件出错')
            try:
                f = open("./data/itemcf.itemusercount.dat", "wb")
                pickle.dump(self.itemusercount, f)
                f.close()
            except Exception as e:
                print('itemcf.itemusercount.dat保存文件出错')

    # t  算法种类
    # 1 -- 传统算法  2 -- 改进算法，性能提高10%-15%
    def fit(self, t=2):
        # 算法分拆成2个函数，方便复用
        try:
            f = open("./data/%s.W.dat" % (self.__class__.__name__,), "rb")
            self.W = pickle.load(f)
            f.close()
            self.calOK = True
        except Exception as e :
            self.calOK = False
            print('%s.W.dat文件不存在' % (self.__class__.__name__,))
        try:
            f = open("./data/%s.iandiuser.dat" % (self.__class__.__name__,), "rb")
            self.iandiuser = pickle.load(f)
            f.close()
            self.calOK = True
        except Exception as e :
            self.calOK = False
            print('%s.iandiuser.dat文件不存在' % (self.__class__.__name__,))
        if self.calOK is False:
            self.__fit(t)
            self.__fitW()
            self.__normalization()
            try:
                f = open("./data/%s.W.dat" % (self.__class__.__name__,), "wb")
                pickle.dump(self.W,f)
                f.close()
            except Exception as e:
                print('%s.W.dat保存文件出错' % (self.__class__.__name__,))
            try:
                f = open("./data/%s.iandiuser.dat" % (self.__class__.__name__,), "wb")
                pickle.dump(self.iandiuser, f)
                f.close()
            except Exception as e:
                print('%s.iandiuser.dat保存文件出错' % (self.__class__.__name__,))

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
            # print(v)
            for kk, vv in v.items():
                self.W[k][kk] = vv / maxW
            # print(self.W[k])

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

    '''
        评测函数
        '''

    def RecallandPrecision(self, N, K):
        """ 计算推荐结果的召回率,准确率
            @param N     推荐结果的数目
            @param K     选取近邻的数目
        """
        hit = 0
        n_recall = 0
        n_precision = 0
        for user in self.train.keys():
            if user in self.test:
                tu = self.test[user]
                rank = self.recommend(user, N, K)
                for item, pui in rank:
                    if item in tu:
                        hit += 1
                n_recall += len(tu)
                n_precision += N
        # print(hit)
        # print(n_recall, n_precision)
        return hit / (n_recall * 1.0), hit / (n_precision * 1.0)

    def Coverage(self, N, K):
        """ 计算推荐结果的覆盖率
            @param N     推荐结果的数目
            @param K     选取近邻的数目
        """
        recommned_items = set()
        all_items = set()

        for user in self.train.keys():
            for item in self.train[user]:
                all_items.add(item)

            rank = self.recommend(user, N, K)
            for item, pui in rank:
                recommned_items.add(item)

        # print('len: ', len(recommned_items), 'all_items:', len(all_items))
        return len(recommned_items) / (len(all_items) * 1.0)

    def Popularity(self, N, K):
        """ 计算推荐结果的新颖度(流行度)
            @param N     推荐结果的数目
            @param K     选取近邻的数目
        """
        item_popularity = dict()
        for user, items in self.train.items():
            for item in items:
                if item not in item_popularity:
                    item_popularity[item] = 0
                item_popularity[item] += 1

        ret = 0
        n = 0
        for user in self.train.keys():
            rank = self.recommend(user, N, K)
            for item, pui in rank:
                ret += math.log(1 + item_popularity[item])
                n += 1
        ret /= n * 1.0
        return ret

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
    M = 5
    key = 1
    N = 10
    K = [5, 10, 20, 30, 40, 80, 160]
    itemcf.splitdata(M, key)
    itemcf.fit()
    for k in K:
        recall, precision = itemcf.RecallandPrecision(N, k)
        popularity = itemcf.Popularity(N, k)
        coverage = itemcf.Coverage(N, k)

        print('userCF: K: %3d, 召回率: %2.4f%% ,准确率: %2.4f%% ,流行度: %2.4f%%, 覆盖率: %2.4f%% ' %
              (k, recall * 100, precision * 100, popularity * 100, coverage * 100))

    itemcfiuf = itemCFIUF('./data/views.dat')
    itemcfiuf.splitdata(M, key)
    itemcfiuf.fit()
    for k in K:
        recall, precision = itemcf.RecallandPrecision(N, k)
        popularity = itemcf.Popularity(N, k)
        coverage = itemcf.Coverage(N, k)

        print('userCFIIF: K: %3d, 召回率: %2.4f%% ,准确率: %2.4f%% ,流行度: %2.4f%%, 覆盖率: %2.4f%% ' %
              (k, recall * 100, precision * 100, popularity * 100, coverage * 100))


