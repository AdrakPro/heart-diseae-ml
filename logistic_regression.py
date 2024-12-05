import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.n_classes = None

    def softmax(self, z):
        """
        Softmax activation function for multiclass classification.

        Parameters:
        - z (np.ndarray): Raw output values (logits).

        Returns:
        - (np.ndarray): Probability values for each class.
        """
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Numerical stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def init_params(self, n_features, n_classes):
        """
        Initializes weights and bias.

        Parameters:
        - n_features (int): Number of features.
        - n_classes (int): Number of classes.
        """
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros((1, n_classes))

    def compute_cost(self, y, y_hat):
        """
        Computes multiclass cross-entropy.

        Parameters:
        - y (np.ndarray): One-hot encoded labels.
        - y_hat (np.ndarray): Predicted probabilities.

        Returns:
        - (float): Cost function value.
        """
        m = y.shape[0]
        epsilon = 1e-10
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)  # Prevent division by zero
        return -(1 / m) * np.sum(y * np.log(y_hat))

    def train(self, X, y, n_classes):
        """
        Trains the multiclass model.

        Parameters:
        - X (np.ndarray): Feature matrix.
        - y (np.ndarray): One-hot encoded labels.
        - n_classes (int): Number of classes.
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self.n_classes = n_classes
        n_samples, n_features = X.shape

        # Initialize parameters
        self.init_params(n_features, n_classes)

        # Gradient descent
        for i in range(self.num_iterations):
            # Calculate predictions
            z = np.dot(X, self.weights) + self.bias
            y_hat = self.softmax(z)

            # Compute cost
            cost = self.compute_cost(y, y_hat)

            # Gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_hat - y))
            db = (1 / n_samples) * np.sum(y_hat - y, axis=0, keepdims=True)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost:.4f}")

    def predict(self, X):
        """
        Predicts classes for input data.

        Parameters:
        - X (np.ndarray): Feature matrix.

        Returns:
        - (np.ndarray): Predicted labels.
        """
        z = np.dot(X, self.weights) + self.bias
        y_hat = self.softmax(z)
        return np.argmax(y_hat, axis=1)