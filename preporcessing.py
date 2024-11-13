import numpy as np


def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    - X (np.ndarray): Feature matrix of shape (n_samples, n_features)
    - y (np.ndarray): Target vector of shape (n_samples, 1)
    - test_size (float): Proportion of the data to include in the test split.
    - random_state (int): Seed for random number generator for reproducibility.

    Returns:
    - X_train (np.ndarray): Training feature matrix
    - X_test (np.ndarray): Testing feature matrix
    - y_train (np.ndarray): Training target vector
    - y_test (np.ndarray): Testing target vector
    """

    if random_state:
        np.random.seed(random_state)

    # Generate an array of indices of all samples
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    # Split index is position in shuffled list where we separate training
    # data from test data
    split_idx = int(X.shape[0] * (1 - test_size))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def preprocess_data(X):
    """
    Preprocesses the data by handling NaN values in the feature matrix.

    Parameters:
    - X (np.ndarray): Feature matrix with potential NaN values.

    Returns:
    - X_processed (np.ndarray): Feature matrix with NaN values imputed.
    """

    nan_mask = np.isnan(X)

    if np.any(nan_mask):
        # Calculate mean of each feature (ignoring NaNs)
        mean_values = np.nanmean(X, axis=0)
        # Replace NaNs with corresponding feature means
        X[nan_mask] = np.take(mean_values, np.where(nan_mask)[1])

    return X


def standardize(X_train, X_test):
    """
    Standardizes the feature matrix so that each feature has zero mean and
    unit variance, using the statistics from the training set to avoid data
    leakage.

    Parameters:
    - X_train (np.ndarray): Training feature matrix of shape (n_samples_train, n_features)
    - X_test (np.ndarray): Testing feature matrix of shape (n_samples_test, n_features)

    Returns:
    - X_train_standardized (np.ndarray): Standardized training feature matrix
    - X_test_standardized (np.ndarray): Standardized testing feature matrix
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    # To prevent division by zero, add a small epsilon to std where std is zero
    std = np.where(std == 0, 1e-8, std)

    X_train_standardized = (X_train - mean) / std
    X_test_standardized = (X_test - mean) / std

    return X_train_standardized, X_test_standardized
