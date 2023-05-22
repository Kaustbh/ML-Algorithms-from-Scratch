import numpy as np
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression
import matplotlib.pyplot as plt
from sklearn import datasets

X,Y=datasets.make_regression(n_samples=100,n_features=4,noise=20,random_state=5)
X_train ,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=12)

#print(X_test.shape)
fig, ax = plt.subplots()

ax.scatter(X[:,0], Y, label='X0')
ax.scatter(X[:,1], Y, label='X1')
ax.scatter(X[:,2], Y, label='X2')
ax.scatter(X[:,3], Y, label='X3')

# add a legend and axis labels
ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()

model=LinearRegression()
model.fit(X_train,Y_train)
predictions=model.predict(X_test)
print(predictions)
print(Y_test)

def mse(predictions,Y_test):
    
    return np.mean(Y_test-predictions)**2

mse=mse(predictions,Y_test)

print(mse)
 

# y_pred_line = reg.predict(X)
# cmap = plt.get_cmap('viridis')
# fig = plt.figure(figsize=(8,6))
# m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
# m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
# plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')
# plt.show()


# y_pred_line = model.predict(X_test)
# cmap = plt.get_cmap('viridis')
# fig = plt.figure(figsize=(8,6))
# ax.scatter(X[:,0], Y, label='X0')
# ax.scatter(X[:,1], Y, label='X1')
# ax.scatter(X[:,2], Y, label='X2')
# ax.scatter(X[:,3], Y, label='X3')

# plt.plot(X[:,0], y_pred_line, color='black', linewidth=2, label='Prediction')
# plt.plot(X[:,1], y_pred_line, color='black', linewidth=2, label='Prediction')
# plt.plot(X[:,2], y_pred_line, color='black', linewidth=2, label='Prediction')
# plt.plot(X[:,3], y_pred_line, color='black', linewidth=2, label='Prediction')
# plt.show()

