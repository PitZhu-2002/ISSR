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

class svm_undersampling_kmeans_distance:
    def __init__(self,data,bootstrap = True,cluster = 3,random_state = 10):
        # 传入 1 必须是少数类
        # data : prime
        # 是否有放回采样
        maj = data[data['label']==0]
        min = data[data['label']==1]
        self.data = data
        if bootstrap == True:
            maj = maj.sample(len(maj),replace=True)#,random_state=random_state)
            min = min.sample(len(min),replace=True)#,random_state=random_state)
            data = maj.append(min)
            #data = data.sample(len(data),replace = True,random_state = random_state + 5)
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
        self.ratio = round(len(self.maj_id) / len(self.minority))
        self.cluster = cluster

    def split(self,n, data):
        ''' 平均划分data ,每份数量
            Input n : 均等划分的数量     data: 聚类某一簇
            Return 均等划分数量列表
        '''
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
            else:
                compile[idx].extend(cluster[a:a + i])
            a = a + i
            idx = idx + 1
        ''' 返回划分好的每部分的数据'''
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
            else:
                compile = self.disk(cluster, spt, compile, judge=1)
        return compile

    def classify(self,cluster):
        # cluster 表示 最终聚类的所有结果
        label = list(set(cluster)) # label 表示 K-means 聚出的簇类别
        label.sort()
        csf = [[] for i in range(len(label))]
        for i in range(len(cluster)):
            lb = cluster[i]
            csf[lb].append(i)
        # 返回各簇在原数据的地址
        return csf

    def KMeans_classify(self):
        X = self.data.iloc[self.maj_id].iloc[:,:-1]
        kmeans = KMeans(n_clusters = self.cluster).fit(X)
        csf = self.classify(kmeans.labels_) # 在原数据的地址
        return csf


    def equal_split(self,csf):
        # 等分的数据集
        # 返回 (按簇等分好的多数类 + 少数类)
        #maj = self.majority
        mnt = self.minority
        idx_set = self.average_cluster(csf)   # 平均划分后的数据
        #data = list(map(lambda d:d.append(mnt),data))  #优化的代码
        # idx_set 表示各个部分的下标集合
        return idx_set

    def support_vector(self,csf):
        # data: (1:1) 数据集 D 最后一列是 label
        # dif 是 已有用 sv 组成的少数类 和 多数类的差值
        # judge : 数量是否有 majority多
        # size: 差值
        #print(csf)
        for d in csf:
            shuffle(d)
        sup_compile = []  # 存储 support vector
        term = 0
        eq_sp = self.equal_split(csf)
        for i in range(self.ratio):
            idx = np.array(eq_sp[i])
            data = self.majority.iloc[idx].append(self.minority)
            # 硬间隔
            svm = SVC(kernel='linear', C=1000000, probability=True, random_state=10)
            svm.fit(data.iloc[:, :-1], data.iloc[:, -1])

            term = term + 1
if __name__ == '__main__':
    data = pd.read_excel('ecoli1.xlsx')#.set_index('代码')
    print('原始数据:',data.iloc[:,-1].value_counts())
    su = svm_undersampling_kmeans_distance(data)

