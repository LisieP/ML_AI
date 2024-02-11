import numpy as np

def accuracy(Y, Y_hat):
    """
    Calculates the accuracy between one-hot encoded
    target vectors (Y), and predictions (Y_hat).

    Parameters:
    -----------
        Y : np.ndarray
            - One hot ground truth (N,K)
        Y_hat : np.array
            - Probabilities (N,K)

    Returns:
    --------
        Percentage number of correct predictions 
    """

    hits = np.equal(np.argmax(Y_hat, -1), np.argmax(Y, -1)).astype("int")

    return np.sum(hits) / Y_hat.shape[0]