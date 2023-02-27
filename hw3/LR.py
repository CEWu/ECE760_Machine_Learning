import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import pdb

dataset = pd.read_csv('emails.csv', header=None, delimiter=',', skiprows=1).iloc[:,1:].to_numpy()
X = dataset[:, 0:-1]
y = dataset[:, -1]
data_index = np.arange(5000, dtype=int)
# kf_test = []
kf_test_index = np.array_split(data_index, 5)
avg = []


test_index = data_index[0:1000]
train_index = np.setdiff1d(data_index, test_index)
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, y_train)
y_pred_knn = neigh.predict(X_test)
print("Accuracy: {:.3f}".format(metrics.accuracy_score(y_test, y_pred_knn)))


class LogisticRegression:
    def __init__(self,x,y):      
        self.intercept = np.ones((x.shape[0], 1))  
        self.x = np.concatenate((self.intercept, x), axis=1)
        self.weight = np.zeros(self.x.shape[1])
        self.y = y
         
    #Sigmoid method
    def sigmoid(self, x, weight):
        z = np.dot(x, weight)
        return 1 / (1 + np.exp(-z))
     
    #method to calculate the Loss
    def loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
     
    #Method for calculating the gradients
    def gradient_descent(self, X, h, y):
        return np.dot(X.T, (h - y)) / y.shape[0]
 
     
    def fit(self, lr , iterations):
        for i in range(iterations):
            sigma = self.sigmoid(self.x, self.weight)
             
            loss = self.loss(sigma,self.y)
 
            dW = self.gradient_descent(self.x , sigma, self.y)
             
            #Updating the weights
            self.weight -= lr * dW
 
        return print('fitted successfully to data')
     
    #Method to predict the class label.
    def predict(self, x_new , treshold):
        self.intercept = np.ones((x_new.shape[0], 1)) 
        x_new = np.concatenate((self.intercept, x_new), axis=1)
        result = self.sigmoid(x_new, self.weight)
        result = result >= treshold
        y_pred = np.zeros(result.shape[0])
        for i in range(len(y_pred)):
            if result[i] == True: 
                y_pred[i] = 1
            else:
                continue
                 
        return y_pred


kf1_y_pred = []
i=0  

for test_index in kf_test_index:
    print()
    train_index = np.setdiff1d(data_index, test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    regressor = LogisticRegression(X_train,y_train)
    regressor.fit(0.0001 , 1000000)
    y_pred = regressor.predict(X_test,0.5)
    print("Accuracy: {:.3f}".format(metrics.accuracy_score(y_test, y_pred)))
    print("Precision: {:.3f}".format(metrics.precision_score(y_test, y_pred)))
    print("Recall: {:.3f}".format(metrics.recall_score(y_test, y_pred)))
    # avg.append(1 - metrics.accuracy_score(y_test, y_pred))
    if i == 0:
      kf1_y_pred = y_pred
    i+=1

# print("avg", sum(avg)/len(avg))


test_index = data_index[0:1000]
train_index = np.setdiff1d(data_index, test_index)
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

#fit logistic regression model and plot ROC curve
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_knn)
auc = round(metrics.roc_auc_score(y_test, y_pred_knn), 4)
plt.plot(fpr,tpr,label="kNN (k=5), AUC="+str(auc))

#fit gradient boosted model and plot ROC curve
fpr, tpr, _ = metrics.roc_curve(y_test, kf1_y_pred)
auc = round(metrics.roc_auc_score(y_test, kf1_y_pred), 4)
plt.plot(fpr,tpr,label="Logistic regression , AUC="+str(auc))


plt.xlabel('FPR')
plt.ylabel('TPR')

#add legend
plt.legend()

plt.savefig('2_5.png')