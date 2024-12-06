import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import StratifiedKFold

from com.hdu.数据分析实战.Business_Analysis.Experiment.mode.Function import Function


def Demo(data):
    names = ['速动比率',	'资产负债率','流动比率','产权比率',	'经营活动产生的现金流量净额／流动负债','利息保障倍数A','现金比率','权益乘数','营运资金与借款比','总资产',
    '营业毛利率',	'销售费用率',	'流动资产净利润率A',	'总资产净利润率(ROA)A',	'固定资产净利润率A'	,'资产报酬率A',	'投入资本回报率（ROIC）',	'财务费用率',
    '投资收益率',	'净资产收益率（ROE）A',	'总资产周转率A',	'流动资产周转率A',	'固定资产周转率A',	'存货周转率A',	'应收账款周转率A',	'应付账款周转率A',
    '股东权益周转率A',	'非流动资产周转率A',	'总资产增长率A',	'净利润增长率A',	'固定资产增长率A',	'资本积累率A',	'综合收益增长率',	'每股净资产增长率A',
    '流动资产比率',	'固定资产比率',	'流动负债比率',	'现金资产比率',	'所有者权益比率',	'营运资金比率',	'营业利润占比',	'股东权益比率',	'市盈率（PE）1',
    '市销率（PS）1',	'市现率（PCF）1',	'市净率（PB）'	]
    names = data.columns
    df = pd.DataFrame(
        columns=['indicator','Accuracy', 'F1', 'Auc', 'G-mean', 'Recall', 'Specificity']
    )
    PM = np.array([0]*6)
    ACC = []
    for i in range(len(data.columns[:-1])):
        train = data.iloc[:,[i,-1]]
        X = train.iloc[:,:-1]
        y = train.iloc[:,-1]
        PM = np.array([0] * 6)
        ACC = []
        result = []
        for p in range(0, 3):
            kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=p)
            for train_index, test_index in kf.split(X, y):
                X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
                y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
                base = tree.DecisionTreeClassifier(criterion='entropy',min_samples_split=10,min_samples_leaf=2,random_state=10)
                base.fit(X_train,y_train)
                ppb = Function.proba_predict_minority(base, X_test)
                PM = PM + Function.cal_F1_AUC_Gmean(
                    y_test=y_test,
                    y_pre =base.predict(X_test),
                    prob=Function.proba_predict_minority(base, X_test)
                )
                ACC.append(base.score(X_test, y_test))

        #print(data.columns[i])
        column = ['F1', 'F2','Auc', 'G-mean', 'Recall', 'Specificity']
        result.append(data.columns[:-1][i])
        result.append(sum(ACC) / len(ACC))
        #print("准确率:", sum(ACC) / len(ACC))
        for m in range(0, len(column)):
            result.append(PM[m]/30)
            #print(column[m], ':', PM[m] / 30)
        print(result)
        df = df.append(pd.Series(result),ignore_index=True)
    df.to_excel('属性关联.xls')


data = pd.read_excel('data.xls')#.set_index('代码')

Demo(data)