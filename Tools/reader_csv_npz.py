import numpy as np
def read_mnist_csv(filename):
    """
    Reads the MNIST dataset stored in filename:

    Arguments:
    ----------
        filename: str
            - Filename for dataset.

    Returns:
    --------
        X : np.ndarray
            - Normalised feature vectors
        y : np.ndarray
            - One-hot encoded target class vectors
    """
    X, y = [], []
    with open(filename) as f:
        for line in f:
            # remove end line character and split the CSV data
            clean_line = line.rstrip().split(",")

            # add the features to array
            feats = []
            for x in clean_line[1:]:
                feats.append(float(x))
            X.append(feats)

            # add the targets to array
            one_hot = np.zeros(10)
            one_hot[int(clean_line[0])] = 1
            y.append(one_hot)

    # convert to numpy arrays and normalise training features
    X = np.array(X) / 255
    y = np.array(y)

    return X, y


def read_mnist_npz(filename):
    """
    Reads the MNISTS dataset from a numpy archive.

    Arguments:
    ----------
        filename: str
            - Filename for numpy archive

    Returns:
    --------
        X : np.ndarray
            - Normalised feature vectors
        Y : np.ndarray
            - One-hot encoded target class vectors

    Notes:
    ------
        .npz files are compressed binary formats created using:
            - np.savez_compressed("filename.npz",arr1=arr1,arr2=arr2,...)
        Which can be loaded by:
            - data = np.load(filename)
        Returing a dict with keys -> arr1,arr2,....
            - arr1 = data['arr1']
            - Values in dictionary are np.ndarray
    """

    data = np.load(filename)

    X = data['X']
    Y = data['Y']

    return X, Y