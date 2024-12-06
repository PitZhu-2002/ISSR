import math
import time
from copy import copy

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import tree
from sklearn.cluster import KMeans, DBSCAN, Birch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.utils import shuffle

from com.hdu.数据分析实战.Business_Analysis.Experiment.mode.Function import Function


# 基分类器 平衡数据集

class svm_undersampling_kmeans:
    def __init__(self,data,bootstrap = True,cluster = 3,random_state = 10):
        # 传入 1 必须是少数类
        # data : prime
        # 是否有放回采样
        maj = data[data['label']==0]
        min = data[data['label']==1]
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
        # 少数类
        self.minority = data[data.iloc[:,-1] == 1 ]
        # 采样得的平衡数据
        #self.prime = copy(self.minority)
        self.prime = pd.DataFrame(columns = self.majority.columns)
        # 不平衡整数比例
        self.ratio = round(len(self.majority) / round(len(self.majority) / len(self.minority)))
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
        #print(base)
        return base

    def disk(self,cluster, spt, compile, judge,rd):
        a = 0
        cluster = shuffle(cluster)#,random_state = rd)          # 新加增加点随机性
        for idx,i in enumerate(spt):
            if judge == 0:
                compile.append(cluster[a: a + i])
            else:
                compile[idx] = compile[idx].append(cluster[a:a + i])
            a = a + i
            idx = idx + 1
        ''' 返回划分好的每部分的数据'''
        return compile

    def average_cluster(self,data,rs):
        n = len(data)
        # new
        data = shuffle(data)#,random_state=rs)
        compile = []
        rd = 0
        for idx, cluster in enumerate(data):
            # cluster 表示聚出的一类
            spt = self.split(n, cluster)
            if idx == 0:
                compile = self.disk(cluster, spt, compile, judge=0,rd = rd)  # 分好的每类
            else:
                compile = self.disk(cluster, spt, compile, judge=1,rd = rd)
            rd = rd + 3
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

    def KMeans_classify(self,majority):
        X = majority.iloc[:, :-1]
        y = majority.iloc[:, -1]
        kmeans = KMeans(n_clusters = self.cluster).fit(X)
        csf = self.classify(kmeans.labels_)
        return csf

    def equal_split(self):
        # 等分的数据集
        # 返回 (按簇等分好的多数类 + 少数类)
        maj = self.majority
        mnt = self.minority
        csf = self.KMeans_classify(maj)
        cluster_data = [maj.iloc[i] for i in csf]   # 等价于 上面注释内容
        data = self.average_cluster(cluster_data,rs = len(cluster_data))   # 平均划分后的数据
        print(len(data))
        data = list(map(lambda d:d.append(mnt),data))  #优化的代码
        return data

    def support_vector(self,size,random_state):
        # data: (1:1) 数据集 D 最后一列是 label
        # dif 是 已有用 sv 组成的少数类 和 多数类的差值
        # judge : 数量是否有 majority多
        # size: 差值
        sup_compile = pd.DataFrame(columns = self.majority.columns)  # 存储 support vector
        major = shuffle(self.majority)
        eq_sp = self.equal_split()
        term = 0
        for data in eq_sp:         # 需要检查
            #svm = SVC(kernel='linear',probability=True,random_state = 10)          # C 和 gamma 对 Support Vector影响
            #svm = SVC(kernel='linear', C=10,probability=True,random_state = 10)
            #svm = SVC(kernel='linear', C=100,probability=True,random_state = 10)
            #svm = SVC(kernel='sigmoid', C=1,probability=True,random_state=10)
            #svm = SVC(kernel='sigmoid', C=10,probability=True,random_state=10)
            #svm = SVC(kernel='sigmoid', C=100,probability=True,random_state=10)
            #svm = SVC(kernel='poly', C=1,probability=True,random_state=10)
            #svm = SVC(kernel='poly', C=10,probability=True,random_state=10)
            #svm = SVC(kernel='poly', C=100,probability=True,random_state=10)
            svm =SVC(kernel='rbf', C=10000,probability=True,random_state=10)
            #svm = SVC(kernel='rbf', C=10,probability=True,random_state=10)
            #svm =SVC(kernel='rbf', C=100,probability=True,random_state=10)
            svm.fit(data.iloc[:,:-1],data.iloc[:,-1])
            support_vec_majority = data.iloc[svm.support_][data.iloc[svm.support_]['label'] == 0]
            term = term + 1
            sup_compile = sup_compile.append(support_vec_majority)    # 存储到 sup_compile 里面
        #return data[data['label'] == 0]
        return sup_compile
        # if size >= len(sup_compile):
        #      return sup_compile
        # else:
        #      return sup_compile.sample(size,random_state = random_state)
    def generate(self,random_state = 10):
        mark = 0
        count = 1
        while mark < len(self.minority):
            dif = 2 * len(self.minority) - len(self.prime)
            spvm = self.support_vector(dif,random_state = count)
            self.prime = self.prime.append(spvm)
            mark = mark + len(spvm)
            count = count + 1
        self.prime = self.prime.sample(len(self.minority)).append(self.minority)
        self.prime.iloc[:,-1] = self.prime.iloc[:,-1].astype(int)






if __name__ == '__main__':
    data = pd.read_excel('data_norm_ori.xls')#.set_index('代码')
    print('原始数据:',data.iloc[:,-1].value_counts())
    su = svm_undersampling_kmeans(data,cluster=20)
    su.generate()
    print(su.prime.iloc[:,-1].value_counts())