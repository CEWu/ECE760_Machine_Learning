from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import datageneration
import pdb

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# CREATE DATASET
objData = datageneration.Data()
X_train, y_train, X_test, y_test, X_val, y_val = objData.generate_dataset1()



def plot_decision_boundary(model, X, y):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.show()

best_k = 1
best_accuracy = 0

for k in range(1, 21):  # Try k from 1 to 20
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    # Check accuracy on validation set
    y_val_pred = knn.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    print(f"k={k}, validation accuracy={accuracy}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

# # Train K-NN with best_k on the full training set and plot decision boundary
# knn = KNeighborsClassifier(n_neighbors=best_k)
# knn.fit(X_train, y_train)
# plot_decision_boundary(knn, X_train, y_train)


# nb = GaussianNB()
# nb.fit(X_train, y_train)
# plot_decision_boundary(nb, X_train, y_train)

# Initialize models
knn = KNeighborsClassifier(n_neighbors=best_k)
nb = GaussianNB()

svm = SVC(kernel='linear', random_state=42)
lr = LogisticRegression(random_state=42)

mu_values = np.arange(1.0, 2.6, 0.2)  # from 1.0 to 2.4 with step size 0.2
test_accuracies_knn = []
test_accuracies_nb = []
test_accuracies_svm = []
test_accuracies_lr = []

for mu in mu_values:
    # Generate new data with different mu
    objData = datageneration.Data();  
    X_train, y_train, X_test, y_test, X_val, y_val = objData.generate_dataset1(mu=mu)
    
    # Training and evaluation
    knn.fit(X_train, y_train)
    nb.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    lr.fit(X_train, y_train)
    
    y_pred_knn = knn.predict(X_test)
    y_pred_nb = nb.predict(X_test)
    y_pred_svm = svm.predict(X_test)
    y_pred_lr = lr.predict(X_test)
    
    test_accuracies_knn.append(accuracy_score(y_test, y_pred_knn))
    test_accuracies_nb.append(accuracy_score(y_test, y_pred_nb))
    test_accuracies_svm.append(accuracy_score(y_test, y_pred_svm))
    test_accuracies_lr.append(accuracy_score(y_test, y_pred_lr))

# Plotting
plt.plot(mu_values, test_accuracies_knn, label='K-NN')
plt.plot(mu_values, test_accuracies_nb, label='Naive Bayes')
plt.plot(mu_values, test_accuracies_svm, label='Linear SVM')
plt.plot(mu_values, test_accuracies_lr, label='Logistic Regression')
plt.xlabel('μ')
plt.ylabel('Test accuracy')
plt.legend()
plt.show()