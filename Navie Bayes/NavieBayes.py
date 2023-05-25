import numpy as np 
import pandas as pd 
import sklearn.datasets 
import seaborn 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix


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

# print(df.shape)

#print(df.head())
X=df.drop(columns=['Target'])
Y=df['Target']
#print(X.head())

#X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=4)

train,test =train_test_split(df,test_size=0.2,random_state=4)

# print(train.shape)

# print(train)



# Calculate P(Y=y) for all possible y

def calculate_prior(df,Y):
        
       # print(len(df))
        classes=sorted(list(df[Y].unique()))
        
       
        
    #   print(df['Target'])
    #    print(Y)
       
        
        
        prior = df['Target'].value_counts()

        prior=prior/len(df)
        
       # print(prior)
       # print(type(prior))

        prior=prior.tolist()

        #print(type(prior))

        # for i in classes:
            
            
        #     prior.append(len(df[df[Y]]==i))/len(df)
        
        return prior
     
   
# Approach 1: Calculate P(X=x|Y=y) using Gaussian dist.

def calculate_likelihood_gaussian(df,feat_name,feat_val,Y,label):

    feat=list(df.columns)

    df = df[df[Y]==label]
    mean,std =df[feat_name].mean(), df[feat_name].std()
    # The formula for Gaussian Density function, derived from Wikipedia, looks like this: (1/√2pi*σ) * exp((-1/2)*((x-μ)²)/(2*σ²)),
    #  where μ is mean, σ² is variance, σ is square root of variance (standard deviation).
       
    p_x_given_y = (1/(np.sqrt(2*np.pi)*std)) * np.exp(-((feat_val-mean)**2/(2*std**2)))

    return p_x_given_y
    

# Calculate P(X=x1|Y=y)P(X=x2|Y=y)...P(X=xn|Y=y) * P(Y=y) 
# for all y and find the maximum

def naive_bayes_gaussian(df,X,Y):
        
    # get feature names

    features = list(df.columns)[:-1]

    # calculate prior

    prior = calculate_prior(df,Y)

    Y_pred = []

    # Loop over every data sample 

    for x in X:

        # calculate likelihood
        # print(x)
        # print("gap")
        labels=sorted(list(df[Y].unique()))
        likelihood = [1]*len(labels)

        for j in range(len(labels)):

            for i in range(len(features)):

                likelihood[j] *= calculate_likelihood_gaussian(df,features[i],x[i],Y,labels[j])
        
        # calculate posterior probability (numerator only)

        post_prob = [1]*len(labels)
        # print(type(post_prob))
        for j in range(len(labels)):

            post_prob[j] = likelihood[j]*prior[j]
           # print(post_prob[j])
        
       # maxi=max(post_prob)
       # Y_pred.append(maxi)
        Y_pred.append(np.argmax(post_prob))
        
    
    return np.array(Y_pred)



X_test= test.iloc[:,:-1].values
Y_test= test.iloc[:,-1].values
Y_pred = naive_bayes_gaussian(train,X=X_test,Y='Target')

# print(Y_pred)
# print(Y_test)

print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print((16+5+8)/len(Y_test)*100)



