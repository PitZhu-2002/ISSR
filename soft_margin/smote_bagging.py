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

class smotebagging:
    def __init__(self,data,bootstrap = True,cluster = 3,random_state = 10):
        # 传入 1 必须是少数类
        # data : prime
        # 是否有放回采样
        maj = data[data['label']==0]
        min = data[data['label']==1]
        if bootstrap == True:
            maj = maj.sample(len(maj),replace = True)
            min = min.sample(len(min),replace = True)
            data = maj.append(min)
        else:
            data = data
        self.X = data.iloc[:,:-1]
        self.y = data.iloc[:,-1]

    def generate(self):
        oversampler = SMOTE(random_state=10)
        X_samp,y_samp = oversampler.fit_resample(self.X,self.y)
        return X_samp,y_samp

