import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import datageneration
from matplotlib.colors import ListedColormap
import pdb

# CREATE DATASET
objData = datageneration.Data()
X_train, y_train, X_test, y_test, X_val, y_val = objData.generate_dataset1()

# Algorithm - sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))

#Loss function - cross entropy
def loss_function(y, X, W):
    return np.mean(-y*np.log(sigmoid(X*W)) - (1-y)*np.log(sigmoid(1-X*W)))

#Optimizer - gradient descent
def gradient_descent(W, X, y):
    return np.dot(X.T, (X*W - y))/y.shape[0]

def weight_update(W, X, y, learning_rate):
    W -= learning_rate*gradient_descent(W, X, y)             
    return W

# Prediction
def predict_prob(X, W):
    return sigmoid(np.dot(X, W))

def predict(X, W, threshold):
    return predict_prob(X, W) >= threshold  

#Combine everything together into a module
class LogisticRegression():
    def __init__(self, 
                 learning_rate = 0.01, 
                 num_iter = 10000, 
                 fit_intercept = True, 
                threshold =0.5):
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.threshold = threshold
        
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0],1))
        return np.concatenate((intercept,X), axis =1)
    
    # Algorithm - sigmoid function
    def __sigmoid(self, z):
        return 1/(1+np.exp(-z))

    #Loss function - cross entropy
    def __loss_function(self, y, h):
        return np.mean(-y*np.log(h) - (1-y)*np.log(h))

    #Optimizer - gradient descent
    def __gradient_descent(self,h, X, y):
        return np.dot(X.T, (h - y))/y.shape[0]

    def __weight_update(self,h, X, y, W, learning_rate):
        W -= learning_rate*self.__gradient_descent(h, X, y)             
        return W    
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        #Init weights W
        self.W = np.zeros(X.shape[1])
        #Find the optimal weights   
        for i in range(self.num_iter):
            z = np.dot(X,self.W )
            h = self.__sigmoid(z)
#             loss_value = self.__loss_function(y, h)
            self.W = self.__weight_update(h, X, y, self.W, self.learning_rate)
            
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        return self.__sigmoid(np.dot(X, self.W))

    def predict(self, X):
        return (self.predict_prob(X) >= self.threshold)*1


#Init a logistic regression model 
cl = LogisticRegression()
#Fit the data with the initialized model
cl.fit(X_train,y_train)
#Do prediction
preds = cl.predict(X_test)

#Accuracy
print((preds==y_test).mean())

markers = ('s', 'x')
colors = ('red', 'blue')
target_names = ['1', '-1']

# -----------------------------------------------------
# Plot the decision boundary
cmap = ListedColormap(colors[:len(np.unique(y_test))])
x1_min, x1_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
x2_min, x2_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
resolution = 0.01 # step size in the mesh
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                        np.arange(x2_min, x2_max, resolution))
Z = cl.predict(np.c_[xx1.ravel(), xx2.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx1.shape)
plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
# plot class samples    
for idx, cl in enumerate(np.unique(y_test)):
    plt.scatter(x= X_test[y_test == cl, 0], 
                y= X_test[y_test == cl, 1],
                alpha= 0.8, 
                c= colors[idx],
                marker= markers[idx], 
                label= str(int(cl)), 
                edgecolor= 'black')    
# plt.set_title('Decision boundary of '+ str(title))
plt.xlabel("$x_1$") 
plt.ylabel("$x_2$")
plt.xlim(xx1.min(), xx1.max())
plt.ylim(xx2.min(), xx2.max())
plt.xticks([]); plt.yticks([])
plt.legend(loc='lower left')
plt.show() 