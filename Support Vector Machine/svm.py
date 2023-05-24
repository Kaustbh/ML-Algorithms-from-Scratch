import numpy as np
import pandas as pd
from  sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# For more accuracy change the lambda parameter( starting from 0.1 ):



data=pd.read_csv('D:\Visual Studio Python\Project\ML algorithms from scratch\Support Vector Machine\diabetes.csv')

X=data.drop(columns='Outcome',axis=1)
Y=data['Outcome']

# X=X.values
# Y=Y.values

scaler=StandardScaler()
scaler.fit(X)

X=scaler.transform(X)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)


# X_train=X_train.values
# Y_train=Y_train.values
# X_test=X_test.values
# Y_test=Y_test.values
 



# learning_rate,lambda_param,n_iterns -: are called Hyperparameter 

# weights , bias -: are called Model Parameter


class SVM:

    def __init__(self,learning_rate=0.001,lambda_param=0.01,n_iters=1000):

        self.lr=learning_rate
        self.lambda_param=lambda_param
        self.n_iters=n_iters
        self.weights=None
        self.bias=None
    
    
    
    
    
    def fit(self,X,Y):

        # init weights and bias
        self.m, self.n= X.shape
        self.weights = np.zeros(self.n)
        self.bias =0
        self.X=X
        self.Y=Y

        for i in range(self.n_iters):

            self.update_weights()
    
   
    def update_weights(self):

        # label encoding from {0,1} to {-1,1}

        y_label=np.where(self.Y<=0,-1,1)
        
        #print(X.shape)
      
        for index,x_i in enumerate(self.X):
            
            

            # condition = y(i)*(weights(i)*X(i) + b)

            condition=y_label[index] * (np.dot(x_i,self.weights) - self.bias) >=1
           
           #  Gradient Descent for SVM dJ/dw(partial derivative of cost function w.r.t weights) , dJ/db :
            
            if (condition==True):

                dw=2*self.lambda_param*self.weights
                db=0
            else:

                dw=2*self.lambda_param*self.weights - np.dot(y_label[index],x_i)
                db=y_label[index]

        
        self.weights=self.weights - self.lr*dw

        self.bias = self.bias - self.lr*db


    def predict(self,X):

        output=np.dot(self.weights,X.T) - self.bias

        predicted_labels=np.sign(output)

        y_hat=np.where(predicted_labels<=-1,0,1)

        return y_hat
    

# For more accuracy change the lambda parameter( starting from 0.1 ):

classifier=SVM(0.001,0.5,1000)

classifier.fit(X_train,Y_train)

X_train_prediction=classifier.predict(X_train)

train_accuracy=accuracy_score(Y_train,X_train_prediction)

print(" Training Accuracy" ,train_accuracy)

X_test_prediction=classifier.predict(X_test)

test_accuracy=accuracy_score(Y_test,X_test_prediction)

print(" Testing Accuracy" ,test_accuracy)



    


    
    