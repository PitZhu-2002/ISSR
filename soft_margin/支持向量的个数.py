import pandas as pd
from sklearn.svm import SVC

data = pd.read_excel('vehicle1.xls')
X = data.iloc[:,:-1]
y = data.iloc[:,-1]
#svm1 = SVC(C=1.0,kernel='linear',decision_function_shape='ovo',probability=True)
#svm2 = SVC(C = 10)
svm3 = SVC(C = 1.0,kernel='linear')
#svm1.fit(X,y)
#svm2.fit(X,y)
svm3.fit(X,y)
print(y.value_counts())
#print(svm1.n_support_)
#print(svm2.n_support_)
print(svm3.n_support_)

# 支持向量要变一下