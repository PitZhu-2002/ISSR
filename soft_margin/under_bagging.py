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

class undersampling:
    def __init__(self,data,bootstrap = True,random_state = 10):
        data_copy = copy(data)
        if bootstrap == True:
            data_copy = data.sample(len(data_copy),replace = True)
            #maj = maj.sample(len(maj),replace=True,random_state = random_state)
            #min = min.sample(len(min),replace=True,random_state = random_state)
            #data = maj.append(min)
        else:
            data = data
        self.majority = data_copy[data_copy.iloc[:,-1] == 0]
        self.minority = data_copy[data_copy.iloc[:,-1] == 1]
        self.prime = copy(self.majority)
    def generate(self):
        self.prime = self.prime.sample(len(self.minority),replace=True).append(self.minority)
        self.prime.iloc[:,-1] = self.prime.iloc[:,-1].astype(int)




