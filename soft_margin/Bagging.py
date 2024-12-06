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

class bagging:
    def generate(self,data,bootstrap=True):
        maj = data[data['label'] == 0]
        min = data[data['label'] == 1]
        if bootstrap == True:
            maj = maj.sample(len(maj),replace=True)
            min = min.sample(len(min),replace=True)
            samp = maj.append(min)
            return samp
        else:
            print('Not bootstrap')






