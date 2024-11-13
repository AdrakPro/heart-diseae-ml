import numpy as np


class LogisticRegression:
    """
    Logistic Regression classifier

    Attributes:
    - learning_rate (float): Step size for updating weights during gradient descent.
    - num_iterations (int): Number of iterations to run the gradient descent optimization.
    - weights (np.ndarray): Weights vector for features, initialized as zeros.
    - bias (float): Bias term, initialized to zero.
    """

    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        """
        Sigmoid activation function to map values between 0 and 1.

        Parameters:
        - z (np.ndarray): Input array (linear combination of weights and features).

        Returns:
        - (np.ndarray): Array with sigmoid function applied element-wise.
        """

        return 1 / (1 + np.exp(-z))

    def init_params(self, n):
        """
        Initializes model weights and bias to zeros.

        Parameters:
        - n (int): Number of features in the input data.
        """
        self.weights = np.zeros((n, 1))
        self.bias = 0

    def forward_propagation(self, X):
        """
        Performs forward propagation by calculating the model's predicted probabilities.

        Parameters:
        - X (np.ndarray): Feature matrix of shape (n_samples, n_features)

        Returns:
        - (np.ndarray): Predicted probabilities for each sample.
        """
        z = np.dot(X, self.weights) + self.bias

        return self.sigmoid(z)

    def compute_cost(self, y, y_hat):
        """
        Computes the binary cross-entropy cost function.

        Parameters:
        - y (np.ndarray): Actual target values
        - y_hat (np.ndarray): Predicted probabilities

        Returns:
        - (float): Computed cost.
        """

        epsilon = 1e-10
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
        m = y.shape[0]
        cost = -(1 / m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

        return cost

    def backward_propagation(self, X, y, y_hat):
        """
        Computes gradients for weights and bias using backward propagation.

        Parameters:
        - X (np.ndarray): Feature matrix
        - y (np.ndarray): Actual target values
        - y_hat (np.ndarray): Predicted probabilities

        Returns:
        - dw (np.ndarray): Gradient of weights
        - db (float): Gradient of bias
        """

        m = X.shape[0]
        dw = (1 / m) * np.dot(X.T, (y_hat - y))
        db = (1 / m) * np.sum(y_hat - y)

        return dw, db

    def train(self, X, y):
        """
        Trains the logistic regression model using gradient descent.

        Parameters:
        - X (np.ndarray): Feature matrix
        - y (np.ndarray): Target values
        """

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_features = X.shape[1]
        self.init_params(n_features)

        for i in range(self.num_iterations):
            y_hat = self.forward_propagation(X)
            cost = self.compute_cost(y, y_hat)

            dw, db = self.backward_propagation(X, y, y_hat)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost:.4f}")

    def predict(self, X):
        """
        Predicts binary labels for input samples using a threshold of 0.5.

        Parameters:
        - X (np.ndarray): Feature matrix

        Returns:
        - (np.ndarray): Predicted binary labels (0 or 1) for each sample.
        """

        y_hat = self.forward_propagation(X)

        return (y_hat > 0.5).astype(int)
