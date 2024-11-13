from ucimlrepo import fetch_ucirepo


def load_heart_disease_data():
    """
    Fetches the Heart Disease dataset from the UCI Machine Learning Repository.

    Returns: - X (np.ndarray): Feature matrix of shape (n_samples,
    n_features) - y (np.ndarray): Target vector of shape (n_samples, 1),
    reshaped as a column vector
    """

    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features.to_numpy()  # matrix
    # Reshape to have one column and as many rows as needed
    y = heart_disease.data.targets.to_numpy().reshape(-1, 1)  # vector

    return X, y
