import sys
import os
from tkinter import Y
import numpy as np

# Add Tools directory to path
tools_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Tools'))
sys.path.append(tools_path)

from reader_csv_npz import read_mnist_npz
from evaluation import accuracy

#******************************************************************#
# import the data
test_data, test_targets = read_mnist_npz('data/test.npz')
train_data, train_targets = read_mnist_npz('data/train.npz')

# Confirm dimentions match
print(f'Training data shape',train_data.shape)
print(f'Training targets shape',train_targets.shape)

print(f'Test data shape',test_data.shape)
print(f'Test targets shape',test_targets.shape)

#******************************************************************#
# Function to compute the softmax    
def softmax(X, axis=-1):
    """
    Calculates the softmax activation of an N-dimensional
    vector X.

    Parameters:
    -----------
        X : np.ndarray
            - Input array
        axis : int
            - Indicates the axis which should be normalised.
              Defaults to the last dimension (-1).

    Returns:
    --------
        Vector after softmax activation 
    """

    X_exp = None
    # Y = (N,K)
    #N,D = X.shape()

    X_exp = np.exp(X)/np.sum((np.exp(X)), 1, keepdims=1)

    return X_exp
    #******************************************************************#
# test softmax
x1 = np.array([[1, 2, 3, 6]])
print(softmax(x1))

####################################################################
# Function to compute the log-likelihood
def ll(Y, Y_hat):
    """
    Calculates the log likelihood between one-hot encoded
    target vectors (Y), and predictions (Y_hat).

    Parameters:
    -----------
        Y : np.ndarray
            - One hot ground truth (N,K)
        Y_hat : np.array
            - Probabilities (N,K)

    Returns:
    --------
        LL : np.ndarray
            - Log likelihoods per sample (N)
    """
    LL = None
    LL = (Y*np.log(Y_hat))

    return LL
# test ll
print(ll(x1,x1+x1))


####################################################################
# create the  logistic regression model                

class MultinomialLogisticRegression:
    """Multinomial Logistic regession model

    Attributes:
    -----------
    self.weights : numpy.ndarray of shape (D,K) where D
        is dimensionality of of weight vector (including
        bias) and K is the number of classes

    self.D : int
        - dimesionality of model (including bias)

    self.K : int
        - dimensionality of targets

    Methods:
    --------
        forward : Computes the forward pass

        train_minibatch : Trains the model for one minibatch
                            using gradient descent
    """

    def __init__(self, D, K):
        """
        Constructs the weight matrix

        Parameters:
        -----------
            D : int
                - dimesionality of model (not including bias)
            K : int
                - dimensionality of targets

        Variables:
        ----------
            self.D : int
                - dimesionality of model (including bias)

            self.K : int
                - dimensionality of targets

            self.weights : numpy.ndarray of shape (D,K) where D
                is dimensionality of of weight vector (including
                bias) and K is the number of classes
        """

        self.D = D + 1  # add 1 to size for bias
        self.K = K

        # initialise weights by sampling from the normal distribution
        np.random.seed(73)  # set random seed for reproducibility
        self.weights = np.random.normal(0, 1e-2, (self.D, self.K))

    def forward(self, X):
        """
        Computes the forward pass through the model

        Parameters:
        -----------
            X : np.array of shape (N,D)

        Returns:
        --------
            output : np.array of shape (N,K)
        """

        #  Implement the forward pass             
        # ensure the input array is the correct shape
        if (len(X.shape) != 2) and X.shape[1] != (self.D - 1):
            raise ValueError("Expected shape of (N,D) where N is" +
                             "number of samples and D is: " + str(self.weights.shape[0] - 1) +
                             "but recieved shape: " + str(X.shape))
        
        # add column of ones to X for bias
        X_ = np.concatenate((np.ones((X.shape[0], 1)), X), axis=-1)
        y_hat = softmax(X_@self.weights)
        # print(self.weights.shape)
        # print(X_.shape)

        return(y_hat)
        #******************************************************************#

    def train_minibatch(self, train_X, train_Y, learning_rate=0.01):
        """
        Trains the model given one minibatch using gradient descent
        trained weights are stored in self.weights

        Parameters:
        -----------
            train_X : np.array
                - Training input features of shape (N,D)
            train_Y : np.array
                - Training set one-hot encoded target vectors of shape (N,K)
            learning_rate : float 
                - Learning rate to use in training (alpha)
        """

        ####################################################################
        #  Implement iterative gradient descent using the minibatch training protocol             #

        if (len(train_X.shape) != 2) and train_X.shape[1] != (self.D - 1):
            raise ValueError("Expected shape of (N,D) where N is" +
                             "number of samples and D is: " + str(self.weights.shape[0] - 1) +
                             " but recieved shape: " + str(train_X.shape))

        if (len(train_Y.shape) != 2) and train_Y.shape[1] != (self.K):
            raise ValueError("Expected shape of (N,K) where N is" +
                             "number of samples but recieved shape: " + str(train_Y.shape))

        X_ = np.concatenate((np.ones((train_X.shape[0], 1)), train_X), axis=-1)
        y_hat = self.forward(train_X)
        W = []
        for i in range(10):
            w = ((train_Y[:, i] - y_hat[:, i]).reshape(train_Y.shape[0], 1)*X_)
            W.append(w)
        # print(W)
        W = np.array(W)
        W_ = np.mean(W, axis=1)
        W_ = W_.T
        self.weights += W_*learning_rate
        #******************************************************************#

####################################################################
#        Function to extract minibatch from X,Y       #
def get_minibatch(X, Y, i, batch_size=60):
    minibatch = []
    mini_x = X[i*batch_size:(i+1)*batch_size, :]
    mini_y = Y[i*batch_size:(i+1)*batch_size, :]
    minibatch = ((mini_x, mini_y))
    return minibatch
    #******************************************************************#
# print(get_minibatch(train_data,train_targets,2))
####################################################################
#        Create and train the model                    #

    # """Multinomial Logistic regession model

    # Attributes:
    # -----------
    # self.weights : numpy.ndarray of shape (D,K) where D
    #     is dimensionality of of weight vector (including
    #     bias) and K is the number of classes

    # self.D : int
    #     - dimesionality of model (including bias)

    # self.K : int
    #     - dimensionality of targets

    # Methods:
    # --------
    #     forward : Computes the forward pass


    #     train_minibatch : Trains the model for one minibatch
    #                         utilising gradient descent
lr = MultinomialLogisticRegression(784, 10)


# lr.train_minibatch(train_data,train_targets)

for i in range(31):
    for j in range(0, 1000):
        (X, Y) = get_minibatch(train_data, train_targets, j)
        # print(mini)
        j = j*60
        lr.train_minibatch(X, Y)

Y_hat = lr.forward(test_data)
print(Y_hat.shape)

print(accuracy(test_targets, Y_hat))
# (self,D,K):
