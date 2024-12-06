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

# 删除被选择的样本
# 基分类器 平衡数据集

class svm_undersampling_kmeans_strip:
    def __init__(self,data,bootstrap = True,cluster = 20,random_state = 10):
        maj = data[data['label']==0]    # 多数类样本
        min = data[data['label']==1]    # 少数类样本
        self.data = data
        self.random_state = random_state
        if bootstrap == True:
            #data = data.sample(len(data),replace=True)
            maj = maj.sample(len(maj),replace=True)
            min = min.sample(len(min),replace=True)
            data = maj.append(min)
        else:
            data = data
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
        # n 表示不平衡比例
        # return:
        size = len(data)
        evy = math.floor(size / n)
        minus = size - n * evy
        base = np.array([evy for i in range(0, n)])
        base[:minus] = base[:minus] + 1
        return base

    def disk(self,cluster, spt, compile, judge):
        a = 0
        #cluster = shuffle(cluster)       # 新加增加点随机性
        for idx,i in enumerate(spt):
            if judge == 0:
                compile.append(cluster[a: a + i])
            elif judge == -1:
                compile[idx].extend(cluster[a:])    # [5,7]
            else:
                compile[idx].extend(cluster[a:a + i])
            a = a + i
            idx = idx + 1
        return compile

    def average_cluster(self,data):
        n = self.ratio
        #data = shuffle(data)#,random_state=rs)
        compile = []
        for idx, cluster in enumerate(data):
            # cluster 表示每个簇中的下标
            spt = self.split(n, cluster)
            if idx == 0:
                compile = self.disk(cluster, spt, compile, judge=0)  # 分好的每类
            elif idx == len(data) - 1:
                compile = self.disk(cluster, spt, compile, judge=-1)
            else:
                compile = self.disk(cluster, spt, compile, judge=1)
        return compile
    def equal_split(self, csf):
        idx_set = self.average_cluster(csf)  # 平均划分后的数据
        return idx_set
    def classify(self,cluster):
        # cluster 表示 最终聚类的所有结果
        label = list(set(cluster))
        label.sort()
        csf = [[] for i in range(len(label))]
        for i in range(len(cluster)):
            lb = cluster[i]
            csf[lb].append(i)
        # 返回各簇在原数据的地址
        return csf

    def KMeans_classify(self):
        X = self.data.iloc[self.maj_id].iloc[:, :-1]
        kmeans = KMeans(n_clusters=self.cluster).fit(X)
        csf = self.classify(kmeans.labels_)  # 在原数据的地址
        # print(csf)
        return csf

    def support_vector(self,csf,random_state):
        # data: (1:1) 数据集 D 最后一列是 label
        # dif 是 已有用 sv 组成的少数类 和 多数类的差值
        # judge : 数量是否有 majority多
        # size: 差值
        for d in range(len(csf)):
            random_state = random_state + 1
            csf[d] = shuffle(csf[d],random_state=random_state)
        sup_compile = []  # 存储 support vector
        term = 0
        eq_sp = self.equal_split(csf)
        for i in range(self.ratio):
            idx = np.array(eq_sp[i])
            data = self.majority.iloc[idx].append(self.minority)
            # 硬间隔
            svm = SVC(kernel='rbf', C=10000, probability=True, random_state=10)
            svm.fit(data.iloc[:, :-1], data.iloc[:, -1])
            spv_idx = idx[svm.support_[data.iloc[svm.support_]['label'] == 0]] # 原样本中找出的 spv 的位置
            term = term + 1
            sup_compile.extend(spv_idx)  # 存储到 sup_compile 里面
        for i in range(len((csf))):
            csf[i] = shuffle(list(set(csf[i]) - set(sup_compile)))
        return sup_compile
    def generate(self):
        csf = self.KMeans_classify()
        mark = 0
        count = 1
        select_maj = []
        rd_st = deepcopy(self.random_state)
        while mark < len(self.minority):
            spvm = self.support_vector(csf,random_state=rd_st)
            select_maj.extend(spvm)
            mark = mark + len(spvm)
            count = count + 1
        select_maj = np.array(select_maj)
        self.prime = self.prime.append(self.majority.iloc[select_maj].sample(len(self.minority)))
        self.prime.iloc[:,-1] = self.prime.iloc[:,-1].astype(int)
if __name__ == '__main__':
    data = pd.read_excel('data_norm_ori.xls')#.set_index('代码')
    print('原始数据:',data.iloc[:,-1].value_counts())
    su = svm_undersampling_kmeans_strip(data,bootstrap=False)
    su.generate()


