# demostration of svm calssifier
import pandas as pd

from sklearn import datasets
cancer= datasets.load_breast_cancer()
cancer.data
cancer.feature_names
x= cancer.data
y= cancer.target
type(x)
#pandas help to play with dataset
# splitting the dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

x_train.shape ,x_test.shape
#creating svm
from sklearn import svm
#clf=svm.SVC(kernel='poly')#clf is classifier and svm is used to separate the data
#clf=svm.SVC(kernel='sigmoid')
clf=svm.SVC(kernel='linear')
clf.fit(x_train,y_train)
#prediction of model

y_pred=clf.predict(x_test)

y_test

from sklearn import metrics
result=metrics.confusion_matrix(y_test,y_pred)
print(result)
accuracy=metrics.accuracy_score(y_test,y_pred)
print(accuracy)