import pandas as pd
import numpy as nm


df= pd.read_csv('Social_Network_Ads.csv')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_array = X.values
y_array = y.values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

X_train.shape ,X_test.shape
#creating svm
from sklearn import svm
clf=svm.SVC(kernel='poly')#clf is classifier and svm is used to separate the data
#clf=svm.SVC(kernel='sigmoid')
#clf=svm.SVC(kernel='linear')
clf.fit(X_train,y_train)
#prediction of model

y_pred=clf.predict(X_test)

y_test

from sklearn import metrics
result=metrics.confusion_matrix(y_test,y_pred)
print(result)
accuracy=metrics.accuracy_score(y_test,y_pred)
print(accuracy)
