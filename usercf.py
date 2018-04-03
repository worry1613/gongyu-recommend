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
import random

import math
import json
import pickle
from operator import itemgetter
import numpy as np
import pandas as pd
import datetime
import time
from copy import deepcopy, copy


# noinspection PyBroadException
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
        # item_users 物品id对应的用户id集合表关系
        self.item_users = dict()
        # user_items 用户id对应的物品id集合表关系
        self.user_items = dict()

        # for row in self.df.itertuples():
        #     self.item_users.setdefault(row[2], set())
        #     self.item_users[row[2]].add(row[1])
        #     self.user_items.setdefault(row[1], set())
        #     self.user_items[row[1]].add(row[2])
        # 每个用户的访问总量
        self.useritemcount = dict()
        # self.useritemcount = self.df.groupby(self.df['userid']).size().to_dict()
        self.test = dict()
        self.train = dict()

        try:
            f = open("./data/usercf.train.dat", "rb")
            self.train = pickle.load(f)
            f.close()
            self.calOK = True
            self.user_items = deepcopy(self.train)
        except Exception as e:
            self.calOK = False
            print('usercf.train.dat文件不存在')


        try:
            f = open("./data/usercf.test.dat", "rb")
            self.test = pickle.load(f)
            f.close()
            self.calOK = True
        except Exception as e:
            self.calOK = False
            print('usercf.test.dat文件不存在')
        try:
            f = open("./data/usercf.item_users.dat", "rb")
            self.item_users = pickle.load(f)
            f.close()
            self.calOK = True
        except Exception as e:
            self.calOK = False
            print('usercf.item_users.dat文件不存在')
        try:
            f = open("./data/usercf.useritemcount.dat", "rb")
            self.useritemcount = pickle.load(f)
            f.close()
            self.calOK = True
        except Exception as e:
            self.calOK = False
            print('usercf.useritemcount.dat文件不存在')

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

            for k, v in self.train.items():
                self.useritemcount.setdefault(k, len(v))
            self.user_items = deepcopy(self.train)

            try:
                f = open("./data/usercf.train.dat", "wb")
                pickle.dump(self.train, f)
                f.close()
            except Exception as e:
                print('usercf.train.dat保存文件出错')

            try:
                f = open("./data/usercf.test.dat", "wb")
                pickle.dump(self.test, f)
                f.close()
            except Exception as e:
                print('usercf.test.dat保存文件出错')
            try:
                f = open("./data/usercf.item_users.dat", "wb")
                pickle.dump(self.item_users, f)
                f.close()
            except Exception as e:
                print('usercf.item_users.dat保存文件出错')
            try:
                f = open("./data/usercf.useritemcount.dat", "wb")
                pickle.dump(self.useritemcount, f)
                f.close()
            except Exception as e:
                print('usercf.useritemcount.dat保存文件出错')

    # t  算法种类
    # 1 -- 传统算法  2 -- 改进算法，性能提高10%-15%
    def fit(self, t=2):
        # 算法分拆成2个函数，方便复用
        try:
            f = open("./data/%s.W.dat" % (self.__class__.__name__,), "rb")
            self.W = pickle.load(f)
            f.close()
            self.calOK = True
        except Exception as e:
            self.calOK = False
            print('%s.W.dat文件不存在' % (self.__class__.__name__,))
        try:
            f = open("./data/%s.uanduitem.dat" % (self.__class__.__name__,), "rb")
            self.uanduitem = pickle.load(f)
            f.close()
            self.calOK = True
        except Exception as e:
            self.calOK = False
            print('%s.uanduitem.dat文件不存在' % (self.__class__.__name__,))

        if self.calOK is False:
            self._fit(t)
            self._fitW()
            try:
                f = open("./data/%s.W.dat" % (self.__class__.__name__,), "wb")
                pickle.dump(self.W, f)
                f.close()
            except Exception as e:
                print('%s.W.dat保存文件出错'  % (self.__class__.__name__,))
            try:
                f = open("./data/%s.uanduitem.dat" % (self.__class__.__name__,), "wb")
                pickle.dump(self.uanduitem, f)
                f.close()
            except Exception as e:
                print('%s.uanduitem.dat保存文件出错' % (self.__class__.__name__,))

    def _fit(self, t):
        '''
        计算 用户与用户矩阵
        :param t:    1 -- 传统算法  2 -- 改进算法，性能提高10%-15%
        :return:
        '''
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
        '''
        计算W矩阵
        :return:
        '''
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
        '''
        推荐
        :param user:  用户
        :param k:     取近邻数
        :param n:     推荐结果数
        :return:
        '''
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
    M = 5
    key = 1
    N = 10
    K = [5,10,20,30,40,80,160]
    ucf.splitdata(M, key)
    ucf.fit()
    for k in K:
        recall, precision = ucf.RecallandPrecision(N, k)
        popularity = ucf.Popularity(N, k)
        coverage = ucf.Coverage(N, k)

        print('userCF: K: %3d, 召回率: %2.4f%% ,准确率: %2.4f%% ,流行度: %2.4f%%, 覆盖率: %2.4f%% ' %
              (k, recall*100, precision*100, popularity*100, coverage*100))

    ucfiif = userCFIIF('./data/views.dat')
    ucfiif.splitdata(M, key)
    ucfiif.fit()
    for k in K:
        recall, precision = ucf.RecallandPrecision(N, k)
        popularity = ucf.Popularity(N, k)
        coverage = ucf.Coverage(N, k)

        print('userCFIIF: K: %3d, 召回率: %2.4f%% ,准确率: %2.4f%% ,流行度: %2.4f%%, 覆盖率: %2.4f%% ' %
              (k, recall*100, precision*100, popularity*100, coverage*100))
