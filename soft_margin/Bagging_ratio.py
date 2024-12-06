import numpy as np
import pandas as pd
from deslib.des import METADES
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler,TomekLinks,EditedNearestNeighbours,RepeatedEditedNearestNeighbours,\
    AllKNN,CondensedNearestNeighbour,ClusterCentroids
from sklearn import tree
from sklearn.ensemble import BaggingClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE,RandomOverSampler
from smote_variants import SMOTE_TomekLinks, SMOTE_ENN, Borderline_SMOTE1, Borderline_SMOTE2, ADASYN, AHC, LLE_SMOTE, \
    distance_SMOTE, SMMO, Stefanowski, Safe_Level_SMOTE
from com.hdu.数据分析实战.Business_Analysis.Experiment.mode.Function import Function
from com.hdu.数据分析实战.Business_Analysis.Experiment.mode.支持向量机.MajorityVotingClassifier import \
    MajorityVotingClassifer
from com.hdu.数据分析实战.Business_Analysis.Experiment.mode.支持向量机.Base_ratio import svm_undersampling



def operate(data):
    # 基分类器初始化
    model_tree1 = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2,min_samples_leaf=1
                                              ,random_state=10)
    model_tree2 = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=5, min_samples_split=5
                                              ,random_state=10)
    model_tree3 = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=5, min_samples_split=10
                                              ,random_state=10)
    model_tree4 = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=20, min_samples_split=5,
                                              random_state=10)
    model_tree5 = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=20, min_samples_split=10,
                                              random_state=10)
    model_tree6 = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None, min_samples_split=2,min_samples_leaf=1
                                              ,random_state=10)
    model_tree7 = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=5, min_samples_split=5,
                                              random_state=10)
    model_tree8 = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=5, min_samples_split=10,
                                              random_state=10)
    model_tree9 = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=20, min_samples_split=5,
                                              random_state=10)
    model_tree10 = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=20, min_samples_split=10,
                                               random_state=10)


    pool = [    model_tree1,model_tree2,model_tree3,model_tree4,model_tree5,
                model_tree6, model_tree7, model_tree8,model_tree9, model_tree10     ]


    for base in pool:
        svm_sam = svm_undersampling(data = data)
        svm_sam.generate()  # 生成 Support Vector 样本
        print(svm_sam.prime.iloc[:,-1])
        base.fit(svm_sam.prime.iloc[:,:-1] , svm_sam.prime.iloc[:,-1])

    voting_clf = MajorityVotingClassifer(pool)
    return voting_clf

def start():
    PM_total = []
    svm_PM = np.array([0]*5)

    ACC_total = []
    svm_ACC = []

    name = ['Baseline', 'SMOTE', 'RandomOverSampler', 'SMOTE_TomekLinks', 'SMOTE_ENN', 'Borderline_SMOTE1',
            'ADASYN','RandomUnderSampler','TomekLinks','EditedNearestNeighbours','RepeatedEditedNearestNeighbours'
            ,'AllKNN']
    for i in range(len(name)):
        PM_total.append(np.array([0] * 5))  # 每个下标代表一个 PM
        ACC_total.append([])                # 每个下标代表一个 准确率

    data = pd.read_excel('00-22data_result.xls').set_index('代码')
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    for p in range(0, 3):
        kf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = p)
        for train_index, test_index in kf.split(X, y):
            X_prime, X_test = np.array(X)[train_index], np.array(X)[test_index]
            y_prime, y_test = np.array(y)[train_index], np.array(y)[test_index]

            pool_sampling = []
            pool_classifier = []  # 下标 表示分类器的种类，对应 name 里面过采样方法产生的数据
            pool_prd = []
            pool_ppb = []

            votingclassifier = operate(data.iloc[train_index])

            svm_ppb = Function.proba_predict_minority(votingclassifier, X_test)
            svm_PM = svm_PM + Function.cal_F1_AUC_Gmean(
                y_test=y_test,
                y_pre=votingclassifier.predict(X_test),
                prob=Function.proba_predict_minority(votingclassifier, X_test)
            )
            svm_ACC.append(votingclassifier.score(X_test, y_test))

            # 传统采样方法
            random_state = '(random_state = 10)'
            for m in range(0, len(name)):

                model_tree1 = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None,
                                                          min_samples_split=2, min_samples_leaf=1
                                                          , random_state=10)
                model_tree2 = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=5,
                                                          min_samples_split=5
                                                          , random_state=10)
                model_tree3 = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=5,
                                                          min_samples_split=10
                                                          , random_state=10)
                model_tree4 = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=20,
                                                          min_samples_split=5,
                                                          random_state=10)
                model_tree5 = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=20,
                                                          min_samples_split=10,
                                                          random_state=10)
                model_tree6 = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None,
                                                          min_samples_split=2, min_samples_leaf=1
                                                          , random_state=10)
                model_tree7 = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=5,
                                                          min_samples_split=5,
                                                          random_state=10)
                model_tree8 = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=5,
                                                          min_samples_split=10,
                                                          random_state=10)
                model_tree9 = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=20,
                                                          min_samples_split=5,
                                                          random_state=10)
                model_tree10 = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=20,
                                                           min_samples_split=10,
                                                           random_state=10)

                pool = [model_tree1, model_tree2, model_tree3, model_tree4,
                        model_tree5, model_tree6, model_tree7, model_tree8,
                        model_tree9, model_tree10]

                base = MajorityVotingClassifer(pool)





                if name[m] == 'Baseline':
                    base.fit(X_prime, y_prime)
                elif name[m] in ('SMOTE', 'RandomOverSampler'):
                    sample = eval(name[m] + random_state)
                    X_sam, y_sam = sample.fit_resample(X_prime, y_prime)
                    base.fit(X_sam, y_sam)
                elif name[m] in ('ClusterCentroids','RandomUnderSampler','TomekLinks','EditedNearestNeighbours',
                    'RepeatedEditedNearestNeighbours','AllKNN','CondensedNearestNeighbour'):
                    sample = eval(name[m]+'()')
                    print(sample)
                    X_sam, y_sam = sample.fit_resample(X_prime,y_prime)
                    base.fit(X_sam, y_sam)
                else:
                    sample = eval(name[m] + random_state)
                    X_sam, y_sam = sample.sample(X_prime, y_prime)
                    base.fit(X_sam, y_sam)
                pool_classifier.append(base)
                pool_prd.append(base.predict(X_test))
                pool_ppb.append(Function.proba_predict_minority(base, X_test))
                PM_total[m] = PM_total[m] + Function.cal_F1_AUC_Gmean(
                    y_test=y_test,
                    y_pre=base.predict(X_test),
                    prob=Function.proba_predict_minority(base, X_test)
                )
                ACC_total[m].append(base.score(X_test, y_test))


    columns = ['F1', 'Auc', 'G-mean', 'Recall', 'Specificity']
    df = pd.DataFrame(
        columns = ['Accuracy','F1','Auc','G-mean','Recall','Specificity']
    )
    print('SVM')
    print("准确率:", sum(svm_ACC) / len(svm_ACC))
    for i in range(0, len(columns)):
        print(columns[i], ':', svm_PM[i] / 30)
    for i in range(0,len(name)):
        data = []
        data.append(np.mean(ACC_total[i]))
        for j in range(0,len(df.columns)-1):
                data.append(PM_total[i][j] / (30))
        df.loc[name[i]] = data
    df.to_excel('不平衡比例3.xls')



if __name__ == '__main__':
    start()