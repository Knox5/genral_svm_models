##decision tree calsifier

import pandas as pd

data = pd.read_csv('tres.csv')
data
data.shape
#independent n dependt variable 
x=pd.DataFrame(data[['sepal_length','sepal_width','petal_length','petal_width']])
y=data['species']
type(x)
##splitting the data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
x_train.shape , x_test.shape
#impt
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(x_train,y_train)
from sklearn.tree import DecisionTreeClassifier,plot_tree
import matplotlib.pyplot as mp
mp.figure(figsize=(8,6))
plot_tree(classifier)
mp.show()
#find accuracy and predict the xtest
#creaate decision tree regression
#performancce of model by mean square error 
#decision tree regression with mtpl attributre
