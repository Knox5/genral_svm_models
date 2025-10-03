##decision tree classifier

import pandas as pd
from sklearn.metrics import mean_squared_error as msee


data = pd.read_csv('tres.csv')
data
data.shape
#independent n dependent variable 
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
y_pred = classifier.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Classifier Accuracy:", accuracy)
#creaate decision tree regression
# Suppose we use only one attribute (petal_length) for regression
X_reg = data[['petal_length']]   # independent variable
y_reg = data['sepal_length']     # dependent variable (predicting sepal length from petal length)
# Split for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Create regressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train_reg, y_train_reg)

# Predictions
y_pred_reg = regressor.predict(X_test_reg)
#performancce of model by mean square error 
mse = mean_squared_error(y_test_reg, y_pred_reg)
print("Regression Mean Squared Error:", mse)
#decision tree regression with mtpl attribute
mp.scatter(X_test_reg, y_test_reg, color="blue", label="Actual")
mp.scatter(X_test_reg, y_pred_reg, color="red", label="Predicted")
mp.xlabel("Petal Length")
mp.ylabel("Sepal Length")
mp.legend()
mp.show()
