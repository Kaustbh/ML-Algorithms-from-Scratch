import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from DecisionTree import DecisionTree
data = datasets.load_breast_cancer()
X,Y = data.data , data.target

X_train, X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.2 , random_state=1234)

clf = DecisionTree()
clf.fit(X_train,Y_train)
predictions = clf.predict(X_test)

def accuracy(Y_test,Y_pred):
    return np.sum(Y_test==Y_pred)/ len(Y_test)

acc= accuracy(Y_test,predictions)
print(acc)


