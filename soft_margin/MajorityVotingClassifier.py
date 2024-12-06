import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC


class MajorityVotingClassifer:
    def __init__(self,estimator):
        # estimator 是训练好的
        self.estimator = estimator
    def predict(self,X_test):
        # 预测
        # 输入: 测试样本的属性
        prd_list = []       # 各个分类预测汇总
        r = []              # 多数类投票结果
        for classifier in self.estimator:
            prd = classifier.predict(X_test)
            prd_list.append(prd)
        result = np.array(prd_list).T
        for i in result:
            r.append(np.argmax(np.bincount(i)))
        # 返回: 多数类投票 预测结果
        return np.array(r)


    def score(self,X_test,y_test):
        prd = self.predict(X_test)
        # 返回: 准确率
        return sum(y_test == prd)/len(y_test)

    def fit(self,X_train,y_train):
        # 基分类器 训练
        for i in self.estimator:
            i.fit(X_train,y_train)
    def predict_proba(self,X_test):
        # 多数类投票 置信度
        prd_list = []
        r = []
        for classifier in self.estimator:
            prd = classifier.predict(X_test)
            prd_list.append(prd)
        result = np.array(prd_list).T
        # 顺序是 0 1 的概率
        for i in result:
            pro = [0,0]
            rec = pd.Series(i).value_counts()
            if len(rec) == 1:
                idx = rec.index[0]
                if idx == 1:
                    pro[1] = 1
                if idx == 0:
                    pro[0] = 1
            else:
                idx = rec.index
                for i in idx:
                    pro[i] = rec[i] /(sum(rec))
            r.append(pro)
        return r




if __name__ == '__main__':

    data = pd.read_excel(r'D:\tool\Machine_Learning1\com\hdu\数据分析实战\Business_Analysis\Experiment\mode\项目一_正常-ST预测\Support Vector Under sampling_Bagging 后\实验一：簇的数量的影响\训练数据_sc\train3.xls').iloc[:,1:]
    data2 = pd.read_excel(r'D:\tool\Machine_Learning1\com\hdu\数据分析实战\Business_Analysis\Experiment\mode\项目一_正常-ST预测\Support Vector Under sampling_Bagging 后\实验一：簇的数量的影响\测试数据_sc\test3.xls').iloc[:,1:]
    model_tree1 = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2,min_samples_leaf=1)
    model_tree2 = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=5, min_samples_split=5)
    model_tree3 = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=5, min_samples_split=5)
    model_tree4 = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=5, min_samples_split=10)
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    X_test = data2.iloc[:,:-1]
    y_test = data2.iloc[:,-1]

    #pool = [ model_tree1, model_tree2, model_tree3, model_tree4 ]
    #pool = [tree.DecisionTreeClassifier(random_state=10) for i in range(20)]
    pool = [tree.DecisionTreeClassifier(random_state=10)  for i in range(2)]
    for i in pool:
        i.fit(X,y)
    mvc = MajorityVotingClassifer(pool)
    mvc.predict(X_test)
    skl = BaggingClassifier(base_estimator=tree.DecisionTreeClassifier(random_state=10),n_estimators=2,bootstrap=False,random_state=10)
    skl.fit(X,y)
    predict = mvc.predict(X_test)
    print('---------------------------')
    print(sum(predict == np.array(y_test))/len(predict))
    print(skl.score(X_test,y_test))

