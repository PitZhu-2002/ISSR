import math
import time
from copy import copy, deepcopy

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import tree
from sklearn.cluster import KMeans, DBSCAN, Birch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.utils import shuffle

from com.hdu.数据分析实战.Business_Analysis.Experiment.mode.Function import Function
import warnings
warnings.filterwarnings("ignore")
# 删除被选择的样本
# 基分类器 平衡数据集

class svm_undersampling_non_kmeans:
    def __init__(self,data,bootstrap = True,cluster = 20,random_state = 10):
        maj = data[data['label']==0]    # 多数类样本
        min = data[data['label']==1]    # 少数类样本
        self.data = data
        if bootstrap == True:
            #data = data.sample(len(data),replace=True)
            maj = maj.sample(len(maj),replace=True,random_state = random_state)
            min = min.sample(len(min),replace=True,random_state = random_state)
            data = maj.append(min)
        else:
            data = data
        self.random_state = random_state
        self.X = data.iloc[:,:-1]
        self.y = data.iloc[:,-1]
        # 多数类
        self.majority = data[data.iloc[:,-1] == 0 ]
        self.maj_id = np.arange(0,len(data[data.iloc[:,-1] == 0 ]))   # 表示所有数据的下标
        # 少数类
        self.minority = data[data.iloc[:,-1] == 1 ]
        # 采样得的平衡数据
        self.prime = deepcopy(self.minority)
        # 不平衡整数比例
        self.ratio = math.floor(len(self.maj_id) / len(self.minority))
        self.cluster = cluster

    def split(self,n, data):
        '''
        :param n: 将 data 划分成 n 份
        :param data: 划分数据
        :return: 划分的结果
        举例： 传入 n = 8,data 大小为 1290
        返回: [162,162,161,161,161,161,161,161]
        '''
        size = len(data)
        evy = math.floor(size / n)
        minus = size - n * evy
        base = np.array([evy for i in range(0, n)])
        base[:minus] = base[:minus] + 1
        return base
    def equal_split(self,maj_idx,random_state):
        # maj_idx:多数类的下标，base表示每一部分的数量
        maj_idx = shuffle(maj_idx,random_state=random_state + 15)
        n = math.floor(len(maj_idx) / len(self.minority))
        spt = self.split(n,self.maj_id)
        back = []
        count = 0
        for idx in range(len(spt)):
            if idx == len(spt) - 1:
                back.append(maj_idx[count:])
            else:
                back.append(
                    maj_idx[count:count + spt[idx]]
                )
            count = count + spt[idx]
        return back,n
    def support_vector(self,rest_maj_idx,random_state):
        eq_sp,n = self.equal_split(rest_maj_idx,random_state)
        sup_compile = []
        term = 0
        for i in range(n):
            idx = eq_sp[i]
            print('maj:',len(idx),'min:',len(self.minority))
            data = self.majority.iloc[idx].append(self.minority)
            #svm = SVC(kernel='linear',C=1,probability=True, random_state=10)  # C 和 gamma 对 Support Vector影响
            #svm = SVC(kernel='linear',C=0.1,probability=True, random_state=10)  # C 和 gamma 对 Support Vector影响
            #svm = SVC(kernel='linear', C=10,probability=True,random_state = 10)
            # svm = SVC(kernel='linear', C=100,probability=True,random_state = 10)
            # svm = SVC(kernel='linear', C=10000,probability=True,random_state = 10)
            #svm = SVC(kernel='sigmoid', C=1,probability=True,random_state=10)
            #svm = SVC(kernel='sigmoid', C=0.1,probability=True,random_state=10)
            #svm = SVC(kernel='sigmoid', C=10,probability=True,random_state=10)
            # svm = SVC(kernel='sigmoid', C=10000,probability=True,random_state=10)
            # svm = SVC(kernel='sigmoid', C=100,probability=True,random_state=10)
            #svm = SVC(kernel='poly', C=1, probability=True, random_state=10)
            #svm = SVC(kernel='poly', C=0.1,probability=True,random_state=10)
            svm = SVC(kernel='poly', C=10,probability=True,random_state=10)
            # svm = SVC(kernel='poly', C=100,probability=True,random_state=10)
            # svm = SVC(kernel='poly', C=10000,probability=True,random_state=10)
            #svm = SVC(kernel='rbf', C=1, probability=True, random_state=10)
            #svm = SVC(kernel='rbf', C=0.1, probability=True, random_state=10)
            #svm = SVC(kernel='rbf', C=10,probability=True,random_state=10)
            # svm = SVC(kernel='rbf', C=100,probability=True,random_state=10)
            # svm =SVC(kernel='rbf', C=10000,probability=True,random_state=10)
            svm.fit(data.iloc[:, :-1], data.iloc[:, -1])
            first_idx = idx[svm.support_[data.iloc[svm.support_]['label'] == 0]]  # 原样本中找出的 spv 的位置
            first_support_vector = deepcopy(self.majority).iloc[first_idx,:-1]
            spv_idx =  first_idx[svm.predict(first_support_vector) == 0]
            term = term + 1
            sup_compile.extend(spv_idx)  # 存储到 sup_compile 里面
        rest_maj_idx = np.sort(list(set(rest_maj_idx) - set(sup_compile)))
        print('rest_数量:',len(rest_maj_idx))
        return sup_compile
    def concat(self,data):
        # Function:
        # A = [[1,2],[3,4,5,6],[7,8]]
        # return back = [1,2,3,4,5,6,7,8]
        back = []
        for d in data:
            back.extend(d)
        return np.array(back)
    def generate(self):
        csf = deepcopy(self.maj_id)     # 复制 self.majority 的顺序
        mark = 0                        # 记录已过滤出的 Support Vector 的个数
        count = 1                       # 记录轮次的 可以删除
        select_maj = []                 # 存储 过滤出的 Support Vector 的位置
        # 问题: 按照循环聚类 末尾的问题
        rd_st = deepcopy(self.random_state)
        while mark < len(self.minority):
            csf = shuffle(csf,random_state = rd_st)
            spvm = self.support_vector(csf,random_state=rd_st)     #
            a = np.sort(spvm)
            select_maj.extend(spvm)
            mark = mark + len(spvm)
            rd_st = rd_st + 1
            count = count + 1
        select_maj = np.array(select_maj)
        self.prime = self.prime.append(self.majority.iloc[select_maj].sample(len(self.minority)))
        self.prime.iloc[:,-1] = self.prime.iloc[:,-1].astype(int)
        print('********')
if __name__ == '__main__':
    data = pd.read_excel('data.xls')#.set_index('代码')
    maj = data[data['label'] == 0]
    min = data[data['label'] == 1]
    #maj = shuffle(maj)
    #data = maj.append(min).iloc[:,1:]
    data.index = [i for i in range(len(data))]
    #print(data)
    #print(len(data.columns))
    #print('原始数据:',data.iloc[:,-1].value_counts())
    su = svm_undersampling_non_kmeans(data,bootstrap=False)
    #su.equal_split(su.maj_id)
    su.generate()





