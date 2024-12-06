import math
from copy import copy, deepcopy

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.utils import shuffle

from com.hdu.数据分析实战.Business_Analysis.Experiment.mode.Function import Function


# 基分类器 平衡数据集

class svm_undersampling:
    '''

    '''
    def __init__(self,data,random_state=10,bootstrap=False):
        # 传入 1 必须是少数类
        # data : prime
        # ① 是否有放回采样
        if bootstrap==True:
            data = data.sample(len(data),replace = True)
        # 样本属性
        self.random_state = random_state
        self.X = data.iloc[:,:-1]
        # 样本标签
        self.y = data.iloc[:,-1]
        #self.svm_minority = pd.DataFrame(columns = self.X.columns[:-1])
        # 多数类样本
        self.majority = data[data.iloc[:,-1] == 0 ]
        # 少数类样本
        self.minority = data[data.iloc[:,-1] == 1 ]
        # minority + svm 采得的样本 -> 得 平衡数据
        self.prime = pd.DataFrame(columns = self.majority.columns)
        # 不平衡比例(取整,在划分 Support Vector 时使用)
        #self.ratio = round(len(self.majority) / round(len(self.majority) / len(self.minority))) # 在 比例小于2是用的
        self.ratio = math.floor(len(self.majority) / len(self.minority)) # 在 比例小于2是用的
        #print(self.ratio)

    '''   作用: 根据 Majority 和 Minority 等分多数类样本    '''
    '''   return: 划分好每段数据量列表                     '''
    '''   举例: 多数类 185 ; 少数类 29
          返回结果: [31,31,31,31,31,30]                  '''
    def split(self,n,data):
        size = len(data)
        evy = math.floor(size / n)
        minus = size - n * evy
        base = np.array([evy for i in range(0, n)])
        base[:minus] = base[:minus] + 1
        #print(base)
        return base
    '''    根据 split 返回对应位置的样本   '''
    def equal_split(self,majority, minority):
        # 等分的数据集
        div = math.floor(len(majority) / len(minority))
        if div < 2:
            return [majority.iloc[i:i + self.ratio].append(minority) for i in range(0, len(majority), self.ratio)]
        else:
            split = self.split(self.ratio,self.majority)
            back = []
            a = 0
            for i in split:
                back.append(majority[a:a+i].append(minority))
                a = a + i
            return back



    def support_vector(self,random_state = 10):
        # data: (1:1) 数据集 D 最后一列是 label
        # dif 是 已有用 sv 组成的少数类 和 多数类的差值
        # judge : 数量是否有 majority多
        sup_compile = pd.DataFrame(columns = self.majority.columns)  # 存储 support vector
        #major = shuffle(self.majority,random_state=random_state)
        major = deepcopy(shuffle(self.majority,random_state=random_state))
        eq_sp = self.equal_split(major, self.minority)
        term = 0
        for data in eq_sp:         # 需要检查
            #svm = SVC(kernel='linear',C=1 ,probability=True, random_state=10)  # C 和 gamma 对 Support Vector影响
            svm = SVC(kernel='linear',C=0.1 ,probability=True, random_state=10)  # C 和 gamma 对 Support Vector影响
            #svm = SVC(kernel='linear', C=10,probability=True,random_state = 10)
            #svm = SVC(kernel='linear', C=100,probability=True,random_state = 10)
            #svm = SVC(kernel='linear', C=10000,probability=True,random_state = 10)
            #svm = SVC(kernel='sigmoid', C=1,probability=True,random_state=10)
            #svm = SVC(kernel='sigmoid', C=10,probability=True,random_state=10)
            #svm = SVC(kernel='sigmoid', C=100,probability=True,random_state=10)
            #svm = SVC(kernel='sigmoid', C=10000,probability=True,random_state=10)
            #svm = SVC(kernel='sigmoid', C=0.1,probability=True,random_state=10)
            #data = shuffle(data,random_state=term)
            #svm = SVC(kernel='poly', C=1,probability=True,random_state=10)
            #svm = SVC(kernel='poly', C=0.1,probability=True,random_state=10)
            #svm = SVC(kernel='poly', C=10,probability=True,random_state=10)
            #svm = SVC(kernel='poly', C=100,probability=True,random_state=10)
            #svm = SVC(kernel='poly', C=10000,probability=True,random_state=10)
            #svm =SVC(kernel='rbf', C=1,probability=True,random_state=10)
            #svm =SVC(kernel='rbf', C=0.1,probability=True,random_state=10)
            #svm = SVC(kernel='rbf', C=10,probability=True,random_state=10)
            #svm = SVC(kernel='rbf', C=100,probability=True,random_state=10)
            #svm =SVC(kernel='rbf', C=10000,probability=True,random_state=10)
            svm.fit(data.iloc[:,:-1],data.iloc[:,-1])
            support_vector = data.iloc[svm.support_]    # 找出的support vector
            support_vec_majority = support_vector[support_vector['label'] == 0] # support vector 中的多数类
            term = term + 1
            #print('第',term,'轮:',len(support_vec_majority))
            sup_compile = sup_compile.append(support_vec_majority)    # 存储到 sup_compile 里面
        # # return : 返回 size 个 support_vector
        return sup_compile

    def generate(self):
        mark = 0
        count = 1
        rd = deepcopy(self.random_state)
        while mark < len(self.minority):
            #dif = 2*len(self.minority) - len(self.prime)
            spvm = self.support_vector(random_state=rd)
            self.prime = self.prime.append(spvm)
            mark = mark + len(spvm)
            rd = rd + 3
            print('epoch:',count)
            count = count + 1
        self.prime = self.prime.sample(len(self.minority)).append(self.minority)
        self.prime.iloc[:,-1] = self.prime.iloc[:,-1].astype(int)






if __name__ == '__main__':
    data = pd.read_excel('yeast-2_vs_4.xls')
    print('原始数据:',data.iloc[:,-1].value_counts())
    su = svm_undersampling(data)
    su.generate()
    print(su.prime.iloc[:,-1].value_counts())



