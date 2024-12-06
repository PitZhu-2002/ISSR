import math
from copy import copy

import numpy
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import f1_score, roc_auc_score, recall_score,precision_score
from sklearn.preprocessing import StandardScaler


class Function:

    @staticmethod
    def normalization(data):
        #scaler = StandardScaler()
        #scaler.fit(data.iloc[:,:-1])
        #print(scaler.transform(data.iloc[:,:-1]))
        #data.iloc[:,:-1] = scaler.transform(data.iloc[:,:-1])
        #pause = copy(data).fillna(0)
        def check(X,mx,mn):
            if pd.isnull(X):
                return X
            else:
                return (X - mn) / (mx - mn)
        nc = data.shape[1] - 1  # len(feature)
        '''
        for i in range(2,nc):
            mx = max(data.iloc[:, i])
            mn = min(data.iloc[:, i])
            data.iloc[:, i] = (data.iloc[:, i] - mn) / (mx - mn)
        '''
        for i in range(0,nc):
            mx = max(data.iloc[:,i].dropna())
            mn = min(data.iloc[:,i].dropna())
            data.iloc[:,i] = data.iloc[:,i].apply(check,args = (mx,mn,))
        return data
    @staticmethod
    def norm2(data):
        nc = data.shape[1] #-1
        # 改了 原来是2
        for i in range(0, nc):

            mx = max(data.iloc[:, i])
            mn = min(data.iloc[:, i])
            data.iloc[:, i] = (data.iloc[:, i] - mn) / (mx - mn)
        return data
    @staticmethod
    def cal_F1_AUC_Gmean(y_test, y_pre, prob):
        '''
        Calculate the F1, AUC and G-Mean of the predicted result
        :param y_test: the true label
        :param y_pre: the predicted label
        :param prob: the probability that the result is predicted to label 1
        :return: the F1, AUC and G-mean results
        '''

        all = []
        y_test = np.array(y_test)
        result = np.array(y_pre)
        f1 = f1_score(y_test.astype(int), result.astype(int),pos_label = 1)
        a = roc_auc_score(y_test, prob)
        precision = precision_score(y_test.astype(int),result.astype(int),pos_label = 1)
        recall = recall_score(y_test.astype(int), result.astype(int),pos_label = 1)
        #precision = recall_score(y_test.astype(int), result.astype(int), pos_label = 0)
        #precision = precision_score(y_test, )
        f2 =  (5*precision*recall)/(4*precision+recall)
        specificity = recall_score(y_test.astype(int), result.astype(int), pos_label=0)
        print('auc:',a)
        #g = math.sqrt(recall * precision)
        g = math.sqrt(recall * specificity)
        #print(specificity==precision)
        all.append(f1)
        all.append(f2)
        all.append(a)
        all.append(g)
        all.append(recall)
        all.append(precision)
        all.append(specificity)
        return np.array(all)

    @staticmethod
    def proba_predict_minority(cls, X_test):
        # 输入 测试集
        # 输入 的 数据集 必须 是 标签为 1 的 是 少数类样本
        # 返回: 对 每一个测试集 预测是 少数类标签 的 概率的列表
        predict_prob = []
        for sample in range(0,len(X_test)):
            #print(X_test)
            predict_prob.append(cls.predict_proba([X_test[sample]])[0][1])  #1
        return np.array(predict_prob)

    @staticmethod
    def fill_df(df):
        clm = df.columns[:-1] # 2:-1
        for nm in clm:
            df.loc[:,nm] = df[nm].fillna(df[nm].mean())
        return df

