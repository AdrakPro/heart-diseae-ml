import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    log_loss,
    roc_auc_score,
)


def softmax(z):
    """
    Softmax activation function for multiclass classification.

    Parameters:
    - z (np.ndarray): Raw output values.

    Returns:
    - (np.ndarray): Probability values for each class.
    """
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def compute_cost(y, y_hat):
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


class MultiClassificationModel:
    def __init__(
        self,
        n_classes,
        learning_rate,
        num_iterations,
        momentum,
        beta2,
        epsilon,
        optimizer,
        batch_size,
    ):
        """
        Initialize the Logistic Regression model.

        Parameters:
        - n_classes (int): Number of model classes
        - learning_rate (float): Learning rate for optimization.
        - num_iterations (int): Number of iterations for training.
        - momentum (float): Momentum parameter (used in Momentum and Adam).
        - beta2 (float): RMSprop parameter for Adam optimizer.
        - epsilon (float): Small constant to avoid division by zero in Adam.
        - optimizer (str): Optimization method ("sgd", "momentum", "adam").
        """
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.momentum = momentum
        self.beta2 = beta2
        self.epsilon = epsilon
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.m_dw = None
        self.v_dw = None
        self.m_db = None
        self.v_db = None
        self.weights = None
        self.bias = None

    # Zamiast 0 uzyć randomowych liczb jako poczatkowe wagi
    def init_params(self, n_features):
        """
        Initializes weights and bias.

        Parameters:
        - n_features (int): Number of features.
        - n_classes (int): Number of classes.
        """
        self.weights = np.zeros((n_features, self.n_classes))
        self.bias = np.zeros((1, self.n_classes))
        self.m_dw = np.zeros((n_features, self.n_classes))
        self.v_dw = np.zeros((n_features, self.n_classes))
        self.m_db = np.zeros((1, self.n_classes))
        self.v_db = np.zeros((1, self.n_classes))

    def fit(self, X, y):
        """
        Trains the multiclass model.

        Parameters:
        - X (np.ndarray): Feature matrix.
        - y (np.ndarray): One-hot encoded labels.
        """
        n_samples, n_features = X.shape
        num_batches = int(np.ceil(n_samples / self.batch_size))
        self.init_params(n_features)

        for i in range(1, self.num_iterations + 1):
            indices = np.arange(n_samples)
            np.random.seed(2137)  # todo add to self
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

            for batch in range(num_batches):
                start = batch * self.batch_size
                end = min(start + self.batch_size, n_samples)
                X_batch = X[start:end]
                y_batch = y[start:end]

                z = np.dot(X_batch, self.weights) + self.bias
                y_hat = softmax(z)

                # Compute gradients
                dw = (1 / X_batch.shape[0]) * np.dot(X_batch.T, (y_hat - y_batch))
                db = (1 / X_batch.shape[0]) * np.sum(
                    y_hat - y_batch, axis=0, keepdims=True
                )

                # Update parameters
                if self.optimizer == "sgd":
                    self.weights -= self.learning_rate * dw
                    self.bias -= self.learning_rate * db
                elif self.optimizer == "momentum":
                    self.m_dw = self.momentum * self.m_dw + (1 - self.momentum) * dw
                    self.m_db = self.momentum * self.m_db + (1 - self.momentum) * db
                    self.weights -= self.learning_rate * self.m_dw
                    self.bias -= self.learning_rate * self.m_db
                elif self.optimizer == "adam":
                    self.m_dw = self.momentum * self.m_dw + (1 - self.momentum) * dw
                    self.m_db = self.momentum * self.m_db + (1 - self.momentum) * db
                    self.v_dw = self.beta2 * self.v_dw + (1 - self.beta2) * (dw**2)
                    self.v_db = self.beta2 * self.v_db + (1 - self.beta2) * (db**2)

                    m_dw_corr = self.m_dw / (1 - self.momentum**i)
                    m_db_corr = self.m_db / (1 - self.momentum**i)
                    v_dw_corr = self.v_dw / (1 - self.beta2**i)
                    v_db_corr = self.v_db / (1 - self.beta2**i)

                    self.weights -= (
                        self.learning_rate
                        * m_dw_corr
                        / (np.sqrt(v_dw_corr) + self.epsilon)
                    )
                    self.bias -= (
                        self.learning_rate
                        * m_db_corr
                        / (np.sqrt(v_db_corr) + self.epsilon)
                    )

            z = np.dot(X, self.weights) + self.bias
            y_hat = softmax(z)
            cost = compute_cost(y, y_hat)

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
        proba = self.predict_proba(X)

        return np.argmax(proba, axis=1)

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return softmax(z)

    def evaluate(self, X, y):
        y_proba = self.predict_proba(X)
        y_pred = self.predict(X)

        accuracy = accuracy_score(y, y_pred)
        conf_matrix = confusion_matrix(y, y_pred)
        class_report = classification_report(y, y_pred, zero_division=0)
        logloss = log_loss(y, y_proba)
        roc_auc = roc_auc_score(y, y_proba, multi_class="ovr")

        return accuracy, conf_matrix, class_report, logloss, roc_auc
