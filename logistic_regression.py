import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    log_loss,
)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class BinaryClassificationModel:
    def __init__(self, learning_rate, num_iterations):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.w = None
        self.b = None
        self.dw = None
        self.db = None

    def init_params(self, dim):
        self.w = np.zeros((dim, 1))
        self.b = 0
        self.dw = np.zeros((dim, 1))
        self.db = 0

    def forward(self, X, y):
        """
        Forward pass to compute activations, gradients, and cost.
        """
        m = X.shape[0]
        A = sigmoid(np.dot(X, self.w) + self.b)  # Compute activations

        # Compute cost
        cost = -(1 / m) * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))

        # Gradients
        self.dw = (1 / m) * np.dot(X.T, (A - y))
        self.db = (1 / m) * np.sum(A - y)

        return cost

    def train(self, X, y):
        """
        Train the model with gradient descent.
        """
        self.init_params(X.shape[1])
        costs = []

        for i in range(self.num_iterations):
            cost = self.forward(X, y)

            self.w -= self.learning_rate * self.dw
            self.b -= self.learning_rate * self.db

            if i % 100 == 0:
                costs.append(cost)
                print(f"Iteration {i}: Cost = {cost:.4f}")

        # Plot the cost function
        plt.plot(np.squeeze(costs))
        plt.ylabel("Cost")
        plt.xlabel("Iterations (per hundreds)")
        plt.title(f"Learning rate = {self.learning_rate}")
        plt.show()

    def predict(self, X):
        """
        Predict the labels for a dataset X.
        """
        m = X.shape[0]
        y_prediction = np.zeros((m, 1))
        A = sigmoid(np.dot(X, self.w) + self.b)

        for i in range(A.shape[0]):
            y_prediction[i, 0] = 1 if A[i, 0] > 0.5 else 0

        return y_prediction

    def evaluate(self, X, y):
        y_proba = sigmoid(np.dot(X, self.w) + self.b)
        y_pred = self.predict(X)

        accuracy = accuracy_score(y, y_pred)
        conf_matrix = confusion_matrix(y, y_pred)
        class_report = classification_report(y, y_pred, zero_division=0)
        logloss = log_loss(y, y_proba)

        return accuracy, conf_matrix, class_report, logloss
