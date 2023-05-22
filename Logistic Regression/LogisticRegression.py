import numpy as np

def sigmoid(X):
        
  return 1/(1+np.exp(-X))

class LogisticRegression():

    
   


    def __init__(self,lr=0.001,n_iters=1000):
        
        self.lr=lr
        self.n_iters=n_iters
        self.weights=None
        self.bias=None

    
   
    

    def fit(self,X,Y):

        n_samples,n_features=X.shape

        self.weights=np.zeros(n_features)
        self.bias=0

        for i in range(self.n_iters):

            pred=np.dot(X,self.weights) + self.bias
            y_pred= sigmoid(pred)

            dw = (1/n_samples)*np.dot(X.T,(y_pred-Y))
            db = (1/n_samples)*np.sum(y_pred-Y)

            self.weights= self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db


    def predict(self,X):

        pred=np.dot(X,self.weights) + self.bias
        y_pred= sigmoid(pred)
        prob = [0 if y<=0.5 else 1 for y in y_pred]
        return prob

        

        