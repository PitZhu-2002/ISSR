from copy import copy

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
    def __init__(self,data):
        # 传入 1 必须是少数类
        # data : prime
        self.X = data.iloc[:,:-1]
        self.y = data.iloc[:,-1]
        self.svm_minority = pd.DataFrame(columns = self.X.columns[:-1])
        self.majority = data[data.iloc[:,-1] == 0 ]
        self.minority = data[data.iloc[:,-1] == 1 ]
        self.prime = copy(self.minority)        # 最终欠采样得的 平衡数据

    def partition(self):
        # balance 存在是 经过 n 次 用于找 Support Vector 的数据
        balance = []
        majo = shuffle(self.majority)
        a = 0                                       # 找出的 Support Vector clustering
        while a < len(self.majority):
            balance.append(
                majo.iloc[a:a+len(self.minority)]   # 一次性找 minority 个
            )
            if a + len(self.minority) > len(self.majority):
                break
            else:
                a = a+len(self.minority)
        print(a)
        print(len(self.majority))
        return balance

    def support_vector(self,data,dif):
        # data: (1:1) 数据集 D 最后一列是 label
        # dif 是 已有用 sv 组成的少数类 和 多数类的差值
        svm = SVC(C = 1)          # C 和 gamma 对 Support Vector影响
        svm.fit(data.iloc[:,:-1],data.iloc[:,-1])
        support_vec_majority = data.iloc[svm.support_][data.iloc[svm.support_]['label'] == 0]
        # return : Minority 在 D 上的 support_vector
        return support_vec_majority.iloc[:dif,:]

    def merge(self,sv_min):
        # input : support_vector
        # prime : 用于 Train 的数据
        # 将 输入的 support_vector 合并到 prime 中
        self.prime = self.prime.append(sv_min)

    def generate(self):
        mark = 0
        while mark < len(self.minority):
            balance = self.partition()      # 每次找 Support Vector 样本的列表
            for sample in balance:
                dif = 2*len(self.minority) - len(self.prime)
                sample_in = sample.append(self.minority)
                spvm = self.support_vector(sample_in,dif)
                self.prime = self.prime.append(spvm)
                mark = mark + len(spvm)





if __name__ == '__main__':

    svm_PM = np.array([0] * 5)
    svm_ACC = []
    ACC = []
    PM = np.array([0] * 5)


    data = pd.read_excel('bupa.xls')

    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    for p in range(0,3):
        kf = StratifiedKFold(n_splits = 10, shuffle=True, random_state = p)
        for train_index, test_index in kf.split(X, y):
            X_prime, X_test = np.array(X)[train_index], np.array(X)[test_index]
            y_prime, y_test = np.array(y)[train_index], np.array(y)[test_index]
            a = svm_undersampling(data.iloc[train_index])
            a.generate()
            smote = SMOTE()


            base_svm = tree.DecisionTreeClassifier(random_state=10)
            base_svm.fit(X_train,y_train)
            base_prime = tree.DecisionTreeClassifier(random_state=10)
            base_prime.fit(X_prime,y_prime)

            #svm_prd = base_svm.predict(X_test)
            svm_ppb = Function.proba_predict_minority(base_svm, X_test)
            svm_PM  = svm_PM + Function.cal_F1_AUC_Gmean(
                y_test=y_test,
                y_pre=base_svm.predict(X_test),
                prob=Function.proba_predict_minority(base_svm, X_test)
            )
            svm_ACC.append(base_svm.score(X_test, y_test))
            # prime
            prime_ppb = Function.proba_predict_minority(base_prime, X_test)
            PM = PM + Function.cal_F1_AUC_Gmean(
                y_test=y_test,
                y_pre=base_prime.predict(X_test),
                prob=Function.proba_predict_minority(base_prime, X_test)
            )
            ACC.append(base_prime.score(X_test, y_test))



    columns = [ 'F1', 'Auc', 'G-mean', 'Recall', 'Specificity']

    print('SVM')

    print("准确率:",sum(svm_ACC)/len(svm_ACC))
    for i in range(0,len(columns)):
        print(columns[i],':',svm_PM[i]/30)

    print('Prime')
    print("准确率:", sum(ACC) / len(ACC))
    for i in range(0, len(columns)):
        print(columns[i], ':', PM[i] / 30)


