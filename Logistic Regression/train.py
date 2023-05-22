import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn import datasets
import matplotlib.pyplot  as plt
from LogisticRegression import LogisticRegression
from sklearn.model_selection import train_test_split

data=datasets.load_breast_cancer()
# print(data.keys())
df=pd.DataFrame(data=data.data,columns=data.feature_names)
df['target']=data.target

# print(df.head())
# print(df.shape)

Y=df['target']
X=df.drop(columns='target')

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=1234,test_size=0.2)

model=LogisticRegression(lr=0.01)
model.fit(X_train,Y_train)
predictions=model.predict(X_test)

# print(predictions[:5])
# print(Y_test[:5])

def accuracy(predictions,Y_test):

    return np.sum(predictions==Y_test)/len(Y_test)

acc=accuracy(predictions,Y_test)

print(acc)
