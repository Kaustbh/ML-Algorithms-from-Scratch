import numpy as np 
import pandas as pd 
import sklearn.datasets 
import seaborn 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Refer notebook for more understanding 

# In the Naive Bayes algorithm, the likelihood represents the probability of observing a particular feature (or attribute) value given a class label.
#  There are different methods for calculating the likelihood in Naive Bayes, depending on the type of data being analyzed. Here are some of the most common methods:
# Gaussian likelihood: This method assumes that the features are continuous and follow a Gaussian (normal) distribution. The likelihood is then calculated as the probability density function (PDF) of the normal distribution with the mean and variance estimated from the training data.
# Multinomial likelihood: This method assumes that the features are discrete and represent counts of occurrences of certain events. The likelihood is then calculated as a multinomial distribution with the probability estimated from the training data.
# Bernoulli likelihood: This method is similar to the multinomial likelihood, but assumes that the features are binary, i.e., they take only two values (0 or 1). The likelihood is then calculated as a Bernoulli distribution with the probability estimated from the training data.
# Kernel density estimation: This method is a non-parametric approach that estimates the likelihood as a kernel density function using the training data. The kernel function can be any continuous probability density function, such as the Gaussian kernel.
# The choice of likelihood calculation method depends on the type of data and the assumptions made about the underlying probability distribution of the data.

data= sklearn.datasets.load_iris()

# print(df.feature_names)
df=pd.DataFrame(data.data,columns=data.feature_names)
df['Target']=pd.Series(data.target)

print(df.shape)

#print(df.head())
X=df.drop(columns=['Target'])
Y=df['Target']
#print(X.head())

#X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=4)

train,test =train_test_split(df,test_size=0.2,random_state=4)

print(train.shape)

print(train)

print(len(df))