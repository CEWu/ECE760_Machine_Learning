import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
import pdb

class Data(object):
    """ Methods surrounding data manipulation."""
   

    def generate_dataset1(self, mu=2.5, seed=1):

        # Define parameters
        cov = np.eye(2)
        n_samples = 750

        # Generate positive and negative class points
        positive_class = np.random.multivariate_normal([mu, 0], cov, n_samples)
        negative_class = np.random.multivariate_normal([-mu, 0], cov, n_samples)

        # Label the data
        positive_labels = np.ones((n_samples, 1))
        negative_labels = -np.ones((n_samples, 1))

        # Combine the positive and negative class points
        X = np.vstack((positive_class, negative_class))
        y = np.vstack((positive_labels, negative_labels))
        y = np.squeeze(y)

        # Randomly create train, validation and test splits
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=250, random_state=seed)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=250, random_state=seed)
        return X_train, y_train, X_test, y_test, X_val, y_val 

    
    def generate_dataset2(self, seed=1):
        X, y = make_circles(n_samples=1500, random_state=seed)
        # Create train, validation, and test splits
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=250, random_state=seed)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=250, random_state=seed)

        return X_train, y_train, X_test, y_test, X_val, y_val
